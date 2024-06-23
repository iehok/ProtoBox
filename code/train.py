import argparse
import glob
import os
import sys
from code.config import Config
from code.utils.data_loader import DataLoader
from code.utils.utils import args_factory, from_json, get_model_class
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW

config = Config()


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        dev_acc_all = float(trainer.callback_metrics.get('dev_acc_all'))
        dev_acc_l10 = float(trainer.callback_metrics.get('dev_acc_l10'))

        save = False
        if dev_acc_all > pl_module.args.best_model_acc_all:
            save = True
        elif dev_acc_all == pl_module.args.best_model_acc_all:
            if dev_acc_l10 > pl_module.args.best_model_acc_l10:
                save = True

        if save:
            pl_module.args.best_model_acc_all = dev_acc_all
            pl_module.args.best_model_acc_l10 = dev_acc_l10
            best_model_name = f'epoch={pl_module.args.current_epoch}.pt'
            pl_module.args.best_ckpt_path = os.path.join(pl_module.args.exp_dir, best_model_name)
            with open(pl_module.args.args_path, 'w') as f:
                yaml.dump(vars(pl_module.args), f, default_flow_style=False)
            checkpoint = {
                'states': pl_module.model.state_dict(),
                'optimizer_states': pl_module.optimizer.state_dict()
            }
            for rm_path in glob.glob(os.path.join(pl_module.args.exp_dir, '*.pt')):
                os.remove(rm_path)
            torch.save(checkpoint, pl_module.args.best_ckpt_path)
            print(f'Model saved at: {pl_module.args.best_ckpt_path}')


class MetricWSD_and_ProtoBox(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_class = get_model_class(self.args)  # CBERTProto or CBERTProtoBox
        self.model = model_class(args=self.args)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)

    def forward(self, batch, mode):
        if mode == 'train':
            return self.model(batch, self.device)
        elif mode == 'val':
            return self.model.forward_wsd(batch, self.device)

    def on_train_epoch_start(self):
        self.args.current_epoch += 1

    def training_step(self, batch, batch_idx):
        loss = self(batch, mode='train')
        self.log_dict({'loss': loss.item()})
        return loss

    def on_validation_epoch_start(self):
        self.golds = []
        self.preds = []

    def validation_step(self, batch, batch_idx):
        pred = self(batch, mode='val')
        example = batch['example']
        gold = example['sensekey']
        self.golds.append(gold)
        self.preds.append(pred)

    def on_validation_epoch_end(self):
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        train_sensekey_to_freq = from_json(os.path.join(train_data_dir, 'sensekey_to_freq.json'))
        iscorrect_all = []
        iscorrect_l10 = []
        for (pred, gold) in zip(self.preds, self.golds):
            iscorrect = (pred == gold)
            iscorrect_all.append(iscorrect)
            if train_sensekey_to_freq[gold] <= 10:
                iscorrect_l10.append(iscorrect)
        dev_acc_all = sum(iscorrect_all) / len(iscorrect_all)
        dev_acc_l10 = sum(iscorrect_l10) / len(iscorrect_l10)
        self.log_dict({'dev_acc_all': dev_acc_all})
        self.log_dict({'dev_acc_l10': dev_acc_l10})

    def configure_optimizers(self):
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        return self.optimizer


def main(args):
    args = args_factory(args)
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    pl_module = MetricWSD_and_ProtoBox(args)
    data_module = DataLoader(args)
    logger = WandbLogger(name=args.run_name, project=config.PROJECT_NAME) if args.wandb else None
    print(f'Trainable params: {sum(p.numel() for p in pl_module.parameters() if p.requires_grad)}')
    print(f'All params      : {sum(p.numel() for p in pl_module.parameters())}')
    trainer = pl.Trainer(
        accelerator='gpu',
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[CheckpointCallback()],
        check_val_every_n_epoch=1,
        devices=args.gpus,
        enable_checkpointing=False,
        log_every_n_steps=10,
        logger=logger,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        num_sanity_val_steps=0,
        precision=args.precision,
        profiler=args.profiler,
        strategy=args.strategy,
    )
    trainer.fit(pl_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--box_dim", type=int, default=128)
    parser.add_argument("--freeze_context_enc", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--max_inference_supports", type=int, default=30)
    parser.add_argument("--model_type", type=str, required=True, help="[cbert-proto | cbert-proto-box]")
    parser.add_argument("--nc", type=int, default=16)
    parser.add_argument("--nq", type=int, default=20)
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--nw", type=int, default=16)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)
