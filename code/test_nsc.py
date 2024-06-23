import argparse
import os
import sys
from code.config import Config
from code.utils.data_loader import DataLoader
from code.utils.utils import from_json, get_model_class, load_args
from collections import defaultdict

import pytorch_lightning as pl
import torch

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
        self.sensekey_to_sims = defaultdict(list)
        self.word_to_thr = {}

    def forward(self, batch):
        return self.model.forward_nsc(batch, self.device)

    def validation_step(self, batch, batch_idx):
        sims, support_sensekeys = self(batch)
        example = batch['example']
        if example['label'] == 0:
            sensekey = example['sensekey']
            sim = sims[0][support_sensekeys.index(sensekey)].item()
            self.sensekey_to_sims[sensekey].append(sim)

    def on_validation_epoch_end(self):
        sensekey_to_thr = {sensekey: sum(sims) / len(sims)
                           for sensekey, sims in self.sensekey_to_sims.items()}
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        train_word_to_sensekeys = from_json(os.path.join(train_data_dir, 'word_to_sensekeys.json'))
        for word, sensekeys in train_word_to_sensekeys.items():
            thr_w = 1e9
            for sensekey in sensekeys:
                if sensekey in sensekey_to_thr:
                    thr_w = min(sensekey_to_thr[sensekey], thr_w)
            if thr_w < 1e9:
                self.word_to_thr[word] = thr_w
        self.thr_default = sum(sensekey_to_thr.values()) / len(sensekey_to_thr)

    def on_test_epoch_start(self):
        self.golds = []
        self.preds = []

    def test_step(self, batch, batch_idx):
        sims, support_sensekeys = self(batch)
        example = batch['example']
        gold = example['label']
        word = example['stem']
        thr = self.word_to_thr.get(word, self.thr_default)
        pred = 1
        for support_sensekey in support_sensekeys:
            sim = sims[0][support_sensekeys.index(support_sensekey)].item()
            if sim > thr:
                pred = 0
        self.golds.append(gold)
        self.preds.append(pred)

    def on_test_epoch_end(self):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for (pred, gold) in zip(self.preds, self.golds):
            if (gold == 1) and (pred == 1):
                tp += 1
            elif (gold == 1) and (pred == 0):
                fn += 1
            elif (gold == 0) and (pred == 1):
                fp += 1
            elif (gold == 0) and (pred == 0):
                tn += 1
        acc = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = precision * recall * 2 / (precision + recall)
        self.log_dict(
            {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        )


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
    data_module = DataLoader(ckpt_args, mode='val')
    trainer.validate(pl_module, data_module)
    data_module = DataLoader(ckpt_args, mode='test')
    results = trainer.test(pl_module, data_module)
    acc, precision, recall, f1 = results[0].values()
    print('acc      :', acc)
    print('precision:', precision)
    print('recall   :', recall)
    print('f1       :', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
