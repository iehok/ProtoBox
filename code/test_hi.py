import argparse
import os
import sys
from code.box.box_wrapper import CenterSigmoidBoxTensor
from code.config import Config
from code.synset_graph import Synset_Graph
from code.utils.data_loader import DataLoader
from code.utils.utils import balanced_sampling, from_json, get_model_class, load_args

import pandas as pd
import pytorch_lightning as pl
import torch
from nltk.corpus import wordnet as wn

config = Config()


class MetricWSD_and_ProtoBox(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        checkpoint = torch.load(self.args.best_ckpt_path, map_location='cpu')
        states = checkpoint['states']
        model_class = get_model_class(self.args)  # CBERTProto or CBERTProtoBox
        self.model = model_class(args=self.args)
        self.model.load_state_dict(states)
        self.box = CenterSigmoidBoxTensor

    def forward(self, batch):
        return self.model.forward_rep(batch, self.device)

    def on_test_epoch_start(self):
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        train_synset_to_sensekeys = from_json(os.path.join(train_data_dir, 'synset_to_sensekeys.json'))
        train_sensekey_to_examples = from_json(os.path.join(train_data_dir, 'sensekey_to_examples.json'))
        train_synsets = list(train_synset_to_sensekeys.keys())
        self.sg = Synset_Graph(self.args.root, train_synsets)

        supports = {}
        for synset, sensekeys in train_synset_to_sensekeys.items():
            examples = [train_sensekey_to_examples[sensekey] for sensekey in sensekeys]
            sexamples, _ = balanced_sampling(examples, self.args.max_inference_supports)
            supports[synset] = sexamples

        self.prototype_reps_tensor = []
        self.synset_to_id = {}
        self.id_to_synset = {}
        for i, (synset, examples) in enumerate(supports.items()):
            batch = self.trainer.datamodule.test_data.collate_fn(examples)
            prototype_rep = self.model.forward_rep(batch, self.device)
            self.prototype_reps_tensor.append(prototype_rep)
            self.synset_to_id[synset] = i
            self.id_to_synset[i] = synset
        self.prototype_reps_tensor = torch.stack(self.prototype_reps_tensor)
        self.prototype_reps = self.box.from_split(self.prototype_reps_tensor)

        if self.args.model_type == 'cbert-proto-box':
            self.synset_to_vol = {}
            vols = self.negative_log_vol(self.prototype_reps).tolist()
            for i, vol in enumerate(vols):
                self.synset_to_vol[self.id_to_synset[i]] = vol

    def on_test_epoch_start(self):
        self.golds = []
        self.preds = []

    def test_step(self, batch, batch_idx):
        example = batch['examples'][0]
        query_sensekey = example['sensekey']
        query_synset = wn.synset_from_sense_key(query_sensekey).name()
        query_rep = self(batch)
        query_rep = self.box.from_split(torch.stack([query_rep]))
        n_max = 10
        thr = 0.5
        self.sg.add_gold(query_synset)
        gold = list(self.sg.G.predecessors(query_synset))
        self.sg.del_node(query_synset)
        hypernyms = self.identify_hypernyms(query_rep, n_max, thr)
        self.golds.append(gold)
        self.preds.append(hypernyms)

    def on_test_epoch_end(self):
        correct = [pred[0] in gold for (pred, gold) in zip(self.preds, self.golds)]
        acc = sum(correct) / len(correct)

        rrs = []
        for (pred, gold) in zip(self.preds, self.golds):
            rr = 0
            for i, v in enumerate(pred):
                if v in gold:
                    rr = 1 / (i + 1)
                    break
            rrs.append(rr)
        mrr = sum(rrs) / len(rrs)

        sims = []
        for (pred, gold) in zip(self.preds, self.golds):
            p = pred[0]
            sim = sum([self.sg.sim(p, g) for g in gold]) / len(gold)
            sims.append(sim)
        wu_palm_sim = sum(sims) / len(sims)

        self.log_dict(
            {
                'acc': acc,
                'mrr': mrr,
                'w&p': wu_palm_sim,
            }
        )

    def negative_log_vol(self, boxes):
        z = boxes.z
        Z = boxes.Z
        eps = (torch.ones(z.shape) * torch.finfo(z.dtype).tiny).to(self.device)
        log_vol = torch.sum(torch.log(Z - z + eps), 1)
        return -log_vol

    def identify_hypernyms(self, query_rep, n_max, thr):
        if self.args.model_type == 'cbert-proto':
            scores = torch.cdist(self.prototype_reps, query_rep).t()
            ids = torch.argsort(scores, dim=1).tolist()[0]
            ids = ids[:n_max]
            hypernyms = [self.id_to_synset[id] for id in ids]
        elif self.args.model_type == 'cbert-proto-box':
            vol = self.negative_log_vol(query_rep).item()
            probs1 = torch.exp(self.model.calc_log_prob(self.prototype_reps, query_rep))
            probs2 = torch.exp(self.model.calc_log_prob(query_rep, self.prototype_reps).t())
            thr1 = thr
            ids1 = torch.where(probs1 > thr1)[1].tolist()
            thr2 = 0.100
            ids2 = torch.where(probs2 < thr2)[1].tolist()
            ids = set(ids1) & set(ids2)
            if ids:
                result = pd.DataFrame()
                for id in ids:
                    synset = self.id_to_synset[id]
                    prob = probs1[0][id].item()
                    diff = vol - self.synset_to_vol[synset]
                    tmp = pd.DataFrame({'synset': [synset], 'prob': [prob], 'diff': [diff]})
                    result = pd.concat([result, tmp])
                result = result[result['diff'] > 0.0]
                result = result.sort_values(['diff'])
                result = result.reset_index(drop=True)
                hypernyms_ = result['synset'].values.tolist()
                if not hypernyms_:
                    hypernyms_ = [self.args.root]
            else:
                hypernyms_ = [self.args.root]
            rank1 = hypernyms_[0]
            id_rank1 = self.synset_to_id[rank1]
            rank1_rep = self.box.from_split(torch.stack([self.prototype_reps_tensor[id_rank1]]))
            probs1 = torch.exp(self.model.calc_log_prob(self.prototype_reps, rank1_rep))
            probs2 = torch.exp(self.model.calc_log_prob(rank1_rep, self.prototype_reps).t())
            scores = (probs1 * probs2 * 2) / (probs1 + probs2)
            sorted, indices = torch.sort(scores, dim=1, descending=True)
            hypernyms = [self.id_to_synset[indices[0][i].item()] for i in range(n_max)]
        return hypernyms


def main(args):
    exp_dir = os.path.join(config.EXP_DIR, args.run_name)
    args_path = os.path.join(exp_dir, 'args.yaml')
    ckpt_args = load_args(args_path)
    ckpt_args.experiment = args.experiment
    pl_module = MetricWSD_and_ProtoBox(ckpt_args)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=False,
        num_nodes=1,
        precision=ckpt_args.precision,
    )
    data_module = DataLoader(ckpt_args, mode='test')
    results = trainer.test(pl_module, data_module)
    acc, mrr, wp = results[0].values()
    print('acc:', acc)
    print('mrr:', mrr)
    print('w&p:', wp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
