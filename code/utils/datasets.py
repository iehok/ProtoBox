import argparse
import os
import random
import sys
from code.config import Config
from code.synset_graph import Synset_Graph
from code.utils.utils import balanced_sampling, from_json, list_from_jsonl

import torch
from nltk.corpus import wordnet as wn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

config = Config()


class TrainData(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer_kwargs = {
            'max_length': self.args.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        self.collate_fn = self.collate_fn1
        self.load_data()
        self.sg = Synset_Graph(args.root, list(self.synset_to_freq.keys()))
        if self.args.model_type == 'cbert-proto':
            self.episodes = self.make_episodes_MetricWSD()
            self.sampling_fnc = self.sampling_MetricWSD
        elif self.args.model_type == 'cbert-proto-box':
            self.all_episodes = self.make_episodes_ProtoBox()
            self.sampling_fnc = self.sampling_ProtoBox

    def __len__(self):
        if self.args.model_type == 'cbert-proto':
            return len(self.episodes)
        elif self.args.model_type == 'cbert-proto-box':
            return len(self.all_episodes[self.args.current_epoch - 1])

    def __getitem__(self, idx):
        return self.sampling_fnc(idx)

    def load_data(self):
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        self.word_to_sensekeys = from_json(os.path.join(train_data_dir, 'word_to_sensekeys.json'))
        self.synset_to_sensekeys = from_json(os.path.join(train_data_dir, 'synset_to_sensekeys.json'))
        self.sensekey_to_examples = from_json(os.path.join(train_data_dir, 'sensekey_to_examples.json'))
        self.synset_to_freq = from_json(os.path.join(train_data_dir, 'synset_to_freq.json'))
        self.sensekey_to_freq = from_json(os.path.join(train_data_dir, 'sensekey_to_freq.json'))
        self.synset_to_hypernyms = from_json(os.path.join(train_data_dir, 'synset_to_hypernyms.json'))

    def make_episodes_MetricWSD(self):
        episodes = []
        for sensekeys in self.word_to_sensekeys.values():
            if len(sensekeys) > 1:
                freqs = [self.sensekey_to_freq[sensekey] for sensekey in sensekeys]
                if sum(freqs) != len(freqs):
                    episodes.append(sensekeys)
        return episodes

    def sampling_MetricWSD(self, idx):
        episode = self.episodes[idx]
        supports = {}
        queries = []
        for sensekey in episode:
            examples = self.sensekey_to_examples[sensekey]
            random.shuffle(examples)
            supports[sensekey] = examples[:self.args.ns]
            qexamples = examples[self.args.ns:]
            if not qexamples:
                if len(supports[sensekey]) >= 2:
                    qexamples = [supports[sensekey].pop()]
            queries.append(qexamples)
        queries, _ = balanced_sampling(queries, self.args.nq)
        batch = [supports, queries]
        return batch

    def make_episodes_ProtoBox(self):
        all_episodes = []
        for i in range(self.args.max_epochs):
            print(f'\r[{i+1}/{self.args.max_epochs}]', end='')
            words = list(self.word_to_sensekeys.keys())
            num_episodes = (len(words) - 1) // self.args.nw + 1
            ok = False
            while not ok:
                ok = True
                random.shuffle(words)
                episodes = []
                for j in range(num_episodes):
                    episode_words = words[self.args.nw * j:self.args.nw * (j + 1)]
                    targets = []
                    others = []
                    for word in episode_words:
                        sensekeys = self.word_to_sensekeys[word]
                        for sensekey in sensekeys:
                            synset = wn.synset_from_sense_key(sensekey).name()
                            targets.append(synset)
                            neighbors = list(self.sg.G.predecessors(synset))
                            if neighbors:
                                others.append(neighbors)
                    targets = list(set(targets))
                    if not others:
                        if sum([self.synset_to_freq[synset] for synset in targets]) == len(targets):
                            ok = False
                            break
                    episode = [targets, others]
                    episodes.append(episode)
            all_episodes.append(episodes)
        print()
        return all_episodes

    def sampling_ProtoBox(self, idx):
        episode = self.all_episodes[self.args.current_epoch - 1][idx]
        targets = episode[0]
        hyponyms = episode[1]

        supports = {}
        queries = []

        for target in targets:
            examples = [self.sensekey_to_examples[sensekey]
                        for sensekey in self.synset_to_sensekeys[target]]
            sexamples, qexamples = balanced_sampling(examples, self.args.ns)
            if not qexamples:
                qexamples = [sexamples.pop()]
            else:
                qexamples = sum(qexamples, [])
            if sexamples:
                supports[target] = sexamples
            queries.append(qexamples)

        queries, _ = balanced_sampling(queries, self.args.nq)

        if len(sum(hyponyms, [])) > self.args.nc - len(supports):
            hyponyms, _ = balanced_sampling(hyponyms, self.args.nc - len(supports))
        else:
            hyponyms = sum(hyponyms, [])

        hyponyms = list(set(hyponyms) - set(targets))

        for hyponym in hyponyms:
            examples = [self.sensekey_to_examples[sensekey]
                        for sensekey in self.synset_to_sensekeys[hyponym]]
            supports[hyponym], _ = balanced_sampling(examples, self.args.ns)

        batch = [supports, queries]
        return batch

    def collate_fn1(self, batch):
        assert len(batch) == 1
        batch = batch[0]
        supports, queries = batch
        return self.create_dual_rep_batch(supports, queries)

    def create_dual_rep_batch(self, supports: dict, queries: list):
        # for support set
        support_ids = []
        support_texts = []
        support_spans = []
        sensekey_to_id = {}
        for i, (sensekey, examples) in enumerate(supports.items()):
            s_texts, s_offsets, s_targetwords, s_pos, s_examplekeys, s_sensekeys, s_spans = \
                list(map(list, zip(*[example.values() for example in examples])))
            s_encoded = self.tokenizer.batch_encode_plus(s_texts, **self.tokenizer_kwargs)
            support_ids.append(s_encoded['input_ids'])
            support_texts.append(s_texts)
            support_spans.append(s_spans)
            sensekey_to_id[sensekey] = i

        # for query set
        query_texts = []
        query_spans = []
        target_ids1 = []
        target_ids2 = []
        for example in queries:
            q_text, q_offset, q_targetword, q_pos, q_examplekey, q_sensekey, q_span = example.values()
            query_texts.append(q_text)
            query_spans.append(q_span)
            if self.args.model_type == 'cbert-proto':
                target_ids1.append(sensekey_to_id[q_sensekey])
            elif self.args.model_type == 'cbert-proto-box':
                q_synset = wn.synset_from_sense_key(q_sensekey).name()
                target_ids1.append([1 if (q_synset == s_synset) or (s_synset in self.synset_to_hypernyms[q_synset]) else 0
                                   for s_synset in supports.keys()])
                target_ids2.append([1 if (q_synset == s_synset) or (q_synset in self.synset_to_hypernyms[s_synset]) else 0
                                   for s_synset in supports.keys()])
        q_encoded = self.tokenizer.batch_encode_plus(query_texts, **self.tokenizer_kwargs)
        query_ids = q_encoded['input_ids']
        target_ids1 = torch.tensor(target_ids1)
        target_ids2 = torch.tensor(target_ids2)

        return {
            'query_ids': query_ids,
            'query_spans': query_spans,
            'support_ids': support_ids,
            'support_spans': support_spans,
            'target_ids1': target_ids1,
            'target_ids2': target_ids2,
        }

    def collate_fn2(self, batch):
        examples = batch
        texts, offsets, targetwords, pos, examplekeys, sensekeys, spans = \
            list(map(list, zip(*[example.values() for example in examples])))
        encoded = self.tokenizer.batch_encode_plus(texts, **self.tokenizer_kwargs)
        ids = encoded['input_ids']
        return {
            'examples': examples,
            'ids': ids,
            'spans': spans,
        }


class DevData(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer_kwargs = {
            'max_length': self.args.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        task_to_collate_fn = {
            'wsd': self.collate_fn1,
            'nsc': self.collate_fn1,
            'hi': self.collate_fn2,
        }
        self.collate_fn = task_to_collate_fn[self.args.experiment]
        self.load_train_data()
        dev_data_dir = os.path.join(self.args.data_dir, 'val')
        self.data = list_from_jsonl(os.path.join(dev_data_dir, f'{self.args.experiment}.jsonl'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_train_data(self):
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        self.train_word_to_sensekeys = from_json(os.path.join(train_data_dir, 'word_to_sensekeys.json'))
        self.train_synset_to_sensekeys = from_json(os.path.join(train_data_dir, 'synset_to_sensekeys.json'))
        self.train_sensekey_to_examples = from_json(os.path.join(train_data_dir, 'sensekey_to_examples.json'))
        self.train_synset_to_freq = from_json(os.path.join(train_data_dir, 'synset_to_freq.json'))
        self.train_sensekey_to_freq = from_json(os.path.join(train_data_dir, 'sensekey_to_freq.json'))
        self.train_synset_to_hypernyms = from_json(os.path.join(train_data_dir, 'synset_to_hypernyms.json'))

    def collate_fn1(self, batch):
        assert len(batch) == 1
        example = batch[0]

        # about query set
        q_text, q_offset, q_targetword, q_pos, q_examplekey, q_sensekey, q_span, *_ = example.values()
        q_encoded = self.tokenizer.encode_plus(q_text, **self.tokenizer_kwargs)
        query_ids = q_encoded['input_ids']

        # about support set
        support_sensekeys = list(self.train_word_to_sensekeys[q_targetword])
        support_ids, support_spans = [], []
        for s_key in support_sensekeys:
            synset = wn.synset_from_sense_key(s_key).name()
            if self.args.model_type == 'cbert-proto':
                examples = self.train_sensekey_to_examples[s_key]
                random.shuffle(examples)
                examples = examples[:self.args.max_inference_supports]
            elif self.args.model_type == 'cbert-proto-box':
                examples = [self.train_sensekey_to_examples[sensekey]
                            for sensekey in self.train_synset_to_sensekeys[synset]]
                examples, _ = balanced_sampling(examples, self.args.max_inference_supports)
            s_texts, s_offsets, s_targetwords, s_pos, s_examplekeys, s_sensekeys, s_spans = \
                list(map(list, zip(*[example.values() for example in examples])))
            support_encoded = self.tokenizer.batch_encode_plus(s_texts, **self.tokenizer_kwargs)
            support_ids.append(support_encoded['input_ids'])
            support_spans.append(s_spans)

        return {
            'example': example,
            'query_ids': query_ids,
            'query_spans': [q_span],
            'support_ids': support_ids,
            'support_spans': support_spans,
            'support_sensekeys': support_sensekeys,
        }

    def collate_fn2(self, batch):
        examples = batch
        texts, offsets, targetwords, pos, examplekeys, sensekeys, spans = \
            list(map(list, zip(*[example.values() for example in examples])))
        encoded = self.tokenizer.batch_encode_plus(texts, **self.tokenizer_kwargs)
        ids = encoded['input_ids']
        return {
            'examples': examples,
            'ids': ids,
            'spans': spans,
        }


class TestData(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer_kwargs = {
            'max_length': self.args.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        task_to_collate_fn = {
            'wsd': self.collate_fn1,
            'nsc': self.collate_fn1,
            'hi': self.collate_fn2,
        }
        self.collate_fn = task_to_collate_fn[self.args.experiment]
        self.load_train_data()
        test_data_dir = os.path.join(self.args.data_dir, 'test')
        self.data = list_from_jsonl(os.path.join(test_data_dir, f'{self.args.experiment}.jsonl'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_train_data(self):
        train_data_dir = os.path.join(self.args.data_dir, 'train')
        self.train_word_to_sensekeys = from_json(os.path.join(train_data_dir, 'word_to_sensekeys.json'))
        self.train_synset_to_sensekeys = from_json(os.path.join(train_data_dir, 'synset_to_sensekeys.json'))
        self.train_sensekey_to_examples = from_json(os.path.join(train_data_dir, 'sensekey_to_examples.json'))
        self.train_synset_to_freq = from_json(os.path.join(train_data_dir, 'synset_to_freq.json'))
        self.train_sensekey_to_freq = from_json(os.path.join(train_data_dir, 'sensekey_to_freq.json'))
        self.train_synset_to_hypernyms = from_json(os.path.join(train_data_dir, 'synset_to_hypernyms.json'))

    def collate_fn1(self, batch):
        assert len(batch) == 1
        example = batch[0]

        # about query set
        q_text, q_offset, q_targetword, q_pos, q_examplekey, q_sensekey, q_span, *_ = example.values()
        q_encoded = self.tokenizer.encode_plus(q_text, **self.tokenizer_kwargs)
        query_ids = q_encoded['input_ids']

        # about support set
        support_sensekeys = list(self.train_word_to_sensekeys[q_targetword])
        support_ids, support_spans = [], []
        for s_key in support_sensekeys:
            synset = wn.synset_from_sense_key(s_key).name()
            if self.args.model_type == 'cbert-proto':
                examples = self.train_sensekey_to_examples[s_key]
                random.shuffle(examples)
                examples = examples[:self.args.max_inference_supports]
            elif self.args.model_type == 'cbert-proto-box':
                examples = [self.train_sensekey_to_examples[sensekey]
                            for sensekey in self.train_synset_to_sensekeys[synset]]
                examples, _ = balanced_sampling(examples, self.args.max_inference_supports)
            s_texts, s_offsets, s_targetwords, s_pos, s_examplekeys, s_sensekeys, s_spans = \
                list(map(list, zip(*[example.values() for example in examples])))
            support_encoded = self.tokenizer.batch_encode_plus(s_texts, **self.tokenizer_kwargs)
            support_ids.append(support_encoded['input_ids'])
            support_spans.append(s_spans)

        return {
            'example': example,
            'query_ids': query_ids,
            'query_spans': [q_span],
            'support_ids': support_ids,
            'support_spans': support_spans,
            'support_sensekeys': support_sensekeys,
        }

    def collate_fn2(self, batch):
        examples = batch
        texts, offsets, targetwords, pos, examplekeys, sensekeys, spans = \
            list(map(list, zip(*[example.values() for example in examples])))
        encoded = self.tokenizer.batch_encode_plus(texts, **self.tokenizer_kwargs)
        ids = encoded['input_ids']
        return {
            'examples': examples,
            'ids': ids,
            'spans': spans,
        }
