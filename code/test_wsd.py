import argparse
import os
import sys
from code.config import Config
from code.utils.data_loader import DataLoader
from code.utils.utils import from_json, get_model_class, load_args

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

    def forward(self, batch):
        return self.model.forward_wsd(batch, self.device)

    def on_test_epoch_start(self):
        self.golds = []
        self.preds = []

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        example = batch['example']
        gold = example['sensekey']
        self.golds.append(gold)
        self.preds.append(pred)

    def on_test_epoch_end(self):
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        train_sensekey_to_freq = from_json(os.path.join(train_data_dir, 'sensekey_to_freq.json'))
        iscorrect_all = []
        iscorrect_l10 = []
        for (pred, gold) in zip(self.preds, self.golds):
            iscorrect = (pred == gold)
            iscorrect_all.append(iscorrect)
            if train_sensekey_to_freq[gold] <= 10:
                iscorrect_l10.append(iscorrect)
        acc_all = sum(iscorrect_all) / len(iscorrect_all)
        acc_l10 = sum(iscorrect_l10) / len(iscorrect_l10)
        self.log_dict({'acc_all': acc_all})
        self.log_dict({'acc_l10': acc_l10})


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
    acc_all, acc_l10 = results[0].values()
    print('acc_all:', acc_all)
    print('acc_l10:', acc_l10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
