import json
import os
import random
from argparse import Namespace
from code.config import Config
from code.models.models import CBERTProto, CBERTProtoBox

import pandas as pd
import yaml

config = Config()


def get_model_class(args):
    if args.model_type == 'cbert-proto':
        return CBERTProto
    elif args.model_type == 'cbert-proto-box':
        return CBERTProtoBox
    else:
        raise ValueError('Model type not implemented.')


def save_args(args):
    with open(args.args_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f'Arg file saved at: {args.args_path}')


def load_args(args_path):
    with open(args_path) as f:
        args = yaml.load(f, Loader=yaml.Loader)
    args = Namespace(**args)
    return args


def args_factory(args):
    args.data_dir = os.path.join(config.DATA_DIR, args.root)
    args.exp_dir = os.path.join(config.EXP_DIR, args.run_name)
    args.args_path = os.path.join(args.exp_dir, 'args.yaml')
    args.experiment = 'wsd'
    args.current_epoch = 0
    args.best_model_acc_all = float('-inf')
    args.best_model_acc_l10 = float('-inf')
    return args


def to_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def balanced_sampling(a: list, n: int):
    b = [c.copy() for c in a]
    random.shuffle(b)
    res = []
    while (b):
        i = 0
        while (i < len(b)):
            res.append(b[i].pop(random.randint(0, len(b[i]) - 1)))
            if not b[i]:
                b.pop(i)
            else:
                i += 1
            if len(res) == n:
                return res, b
    return res, []


def list_from_jsonl(path):
    df = pd.read_json(path, lines=True)
    columns = df.columns.tolist()
    rows = df.values.tolist()
    data = []
    for row in rows:
        d = {k: v for (k, v) in zip(columns, row)}
        data.append(d)
    return data
