import argparse
import os
import sys
import warnings
from code.config import Config
from code.models.context_encoder import get_subtoken_indecies
from code.utils.data_utils import load_data
from code.utils.utils import from_json, to_json
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn
from transformers import BertTokenizerFast

config = Config()


def prepare_data(args, mode):
    max_length = args.max_length
    root = args.root
    data_dir = args.data_dir
    mode_data_dir = os.path.join(data_dir, mode)
    Path(mode_data_dir).mkdir(parents=True, exist_ok=True)

    data = load_raw_data(mode)
    examples, synset_to_hypernyms = get_examples(data, root)
    examples = filtering(examples, max_length)
    to_json(synset_to_hypernyms, os.path.join(mode_data_dir, 'synset_to_hypernyms.json'))

    df = pd.DataFrame(examples)
    df.to_json(os.path.join(mode_data_dir, f'{mode}.jsonl'), force_ascii=False, lines=True, orient='records')

    (
        word_to_sensekeys,
        synset_to_sensekeys,
        sensekey_to_examples,
        word_to_freq,
        synset_to_freq,
        sensekey_to_freq
    ) = make_dict(examples)
    to_json(word_to_sensekeys, os.path.join(mode_data_dir, 'word_to_sensekeys.json'))
    to_json(synset_to_sensekeys, os.path.join(mode_data_dir, 'synset_to_sensekeys.json'))
    to_json(sensekey_to_examples, os.path.join(mode_data_dir, 'sensekey_to_examples.json'))
    to_json(word_to_freq, os.path.join(mode_data_dir, 'word_to_freq.json'))
    to_json(synset_to_freq, os.path.join(mode_data_dir, 'synset_to_freq.json'))
    to_json(sensekey_to_freq, os.path.join(mode_data_dir, 'sensekey_to_freq.json'))

    if mode in ['val', 'test']:
        train_word_to_sensekeys = from_json(os.path.join(data_dir, 'train/word_to_sensekeys.json'))
        train_synset_to_sensekeys = from_json(os.path.join(data_dir, 'train/synset_to_sensekeys.json'))
        wsd_data = pd.DataFrame(get_wsd_data(examples, train_word_to_sensekeys))
        nsc_data = pd.DataFrame(get_nsc_data(examples, train_word_to_sensekeys))
        hi_data = pd.DataFrame(get_hi_data(examples, train_synset_to_sensekeys))
        wsd_data.to_json(os.path.join(mode_data_dir, 'wsd.jsonl'), force_ascii=False, lines=True, orient='records')
        nsc_data.to_json(os.path.join(mode_data_dir, 'nsc.jsonl'), force_ascii=False, lines=True, orient='records')
        hi_data.to_json(os.path.join(mode_data_dir, 'hi.jsonl'), force_ascii=False, lines=True, orient='records')


def load_raw_data(mode):
    if mode == 'train':
        return load_data(*config.SEMCOR)
    elif mode == 'val':
        return load_data(*config.SE07)
    elif mode == 'test':
        return load_data(*config.ALL)


def get_examples(data, root):
    examples = []
    synset_to_hypernyms = {}
    for sent in data:
        original_text_tokens = list(map(list, zip(*sent)))[0]
        for offset, (_, stem, pos, examplekey, sensekey) in enumerate(sent):
            if (pos != 'NOUN') or (sensekey == -1):
                continue
            synset = wn.synset_from_sense_key(sensekey).name()
            if synset not in synset_to_hypernyms:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    hypernyms = list(wn.synset(synset).closure(lambda s: s.hypernyms()))
                hypernyms = list(map(lambda x: x.name(), hypernyms))
                synset_to_hypernyms[synset] = hypernyms
            if (root in synset_to_hypernyms[synset]) or (synset == root):
                text_tokens = original_text_tokens[:]
                text = ' '.join(text_tokens)
                example = {
                    'text': text,
                    'offset': offset,
                    'stem': stem,
                    'pos': pos,
                    'examplekey': examplekey,
                    'sensekey': sensekey
                }
                examples.append(example)
    return examples, synset_to_hypernyms


def filtering(examples, max_length):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer_kwargs = {
        'max_length': max_length,
        'padding': 'max_length',
        'truncation': True,
        'return_offsets_mapping': True,
        'return_tensors': 'pt',
    }
    examples_selected = []
    for example in examples:
        text, offset, stem, pos, examplekey, sensekey = example.values()
        encoded = tokenizer.encode_plus(text, **tokenizer_kwargs)
        text_tokens = text.split()
        start, end = get_subtoken_indecies(text_tokens, encoded['offset_mapping'][0].tolist(), offset)
        if start >= end:
            continue
        example['span'] = [start, end]
        examples_selected.append(example)
    return examples_selected


def make_dict(examples):
    word_to_sensekeys = {}      # {word: [sensekeys]}
    synset_to_sensekeys = {}    # {synset: [sensekeys]}
    sensekey_to_examples = {}   # {sensekey: [examples]}
    word_to_freq = {}           # {word: freq}
    synset_to_freq = {}         # {synset: freq}
    sensekey_to_freq = {}       # {sensekey: freq}
    for example in examples:
        text, offset, stem, pos, examplekey, sensekey, span = example.values()
        synset = wn.synset_from_sense_key(sensekey).name()
        insert_to_list_in_dict(word_to_sensekeys, stem, sensekey)
        insert_to_list_in_dict(synset_to_sensekeys, synset, sensekey)
        insert_to_list_in_dict(sensekey_to_examples, sensekey, example)
    for word, sensekeys in word_to_sensekeys.items():
        freq = sum([len(sensekey_to_examples[sensekey]) for sensekey in sensekeys])
        word_to_freq[word] = freq
    for synset, sensekeys in synset_to_sensekeys.items():
        freq = sum([len(sensekey_to_examples[sensekey]) for sensekey in sensekeys])
        synset_to_freq[synset] = freq
    for sensekey, sexamples in sensekey_to_examples.items():
        sensekey_to_freq[sensekey] = len(sexamples)
    return (
        word_to_sensekeys,
        synset_to_sensekeys,
        sensekey_to_examples,
        word_to_freq,
        synset_to_freq,
        sensekey_to_freq
    )


def insert_to_list_in_dict(d, k, v):
    if k not in d:
        d[k] = [v]
    else:
        if v not in d[k]:
            d[k].append(v)


def get_wsd_data(examples, train_word_to_sensekeys):
    data = []
    for example in examples:
        text, offset, targetword, pos, examplekey, sensekey, span = example.values()
        if targetword in train_word_to_sensekeys:
            s_keys = train_word_to_sensekeys[targetword]
            s_synsets = list(map(lambda s_key: wn.synset_from_sense_key(s_key).name(), s_keys))
            synset_target = wn.synset_from_sense_key(sensekey).name()
            if (synset_target in s_synsets) and (len(s_synsets) >= 2):
                data.append(example)
    return data


def get_nsc_data(examples, train_word_to_sensekeys):
    data = []
    for example in examples:
        text, offset, targetword, pos, examplekey, sensekey, span = example.values()
        if targetword in train_word_to_sensekeys:
            label = {'label': 0 if sensekey in train_word_to_sensekeys[targetword] else 1}
            data.append(example | label)
    return data


def get_hi_data(examples, train_synset_to_sensekeys):
    data = []
    for example in examples:
        text, offset, targetword, pos, examplekey, sensekey, span = example.values()
        synset_target = wn.synset_from_sense_key(sensekey).name()
        if synset_target not in train_synset_to_sensekeys:
            data.append(example)
    return data


def main(args):
    args.data_dir = os.path.join(config.DATA_DIR, args.root)
    for mode in ['train', 'val', 'test']:
        prepare_data(args, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--root", type=str, default='animal.n.01')
    args = parser.parse_args()
    main(args)
