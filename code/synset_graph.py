import os
import sys
from code.utils.utils import from_json

import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import wordnet as wn


class Synset_Graph:
    def __init__(self, root, synsets):
        self.root = root
        self.synsets = synsets
        self.G = nx.DiGraph()
        self.depth = {}
        self._make_graph()
        self._set_depth()

    def _make_graph(self):
        for synset in self.synsets:
            hypernyms = self._recursive_sampling_hypernyms(synset)
            for hypernym in hypernyms:
                self.G.add_edge(hypernym, synset)

    def _recursive_sampling_hypernyms(self, synset):
        res = set()
        hypernyms = wn.synset(synset).hypernyms()
        hypernyms = list(map(lambda x: x.name(), hypernyms))
        for hypernym in hypernyms:
            if hypernym in self.synsets:
                res.add(hypernym)
            else:
                res |= self._recursive_sampling_hypernyms(hypernym)
        return res

    def _set_depth(self):
        for synset in self.synsets:
            self.depth[synset] = nx.shortest_path_length(self.G, source=self.root, target=synset) + 1

    def sim(self, synset1, synset2):
        depth1 = self.depth[synset1]
        depth2 = self.depth[synset2]
        lowest_common_ancestor = nx.lowest_common_ancestor(self.G, synset1, synset2)
        depth3 = self.depth[lowest_common_ancestor]
        wu_palmer = depth3 * 2 / (depth1 + depth2)
        return wu_palmer

    def add_gold(self, new_synset):
        hypernyms = self._recursive_sampling_hypernyms(new_synset)
        for hypernym in hypernyms:
            self.G.add_edge(hypernym, new_synset)
        self.depth[new_synset] = min(
            [self.depth[hypernym] for hypernym in hypernyms]) + 1

    def add_node(self, new_synset, hypernyms=[], hyponyms=[]):
        for hypernym in hypernyms:
            self.G.add_edge(hypernym, new_synset)
        for hyponym in hyponyms:
            self.G.add_edge(new_synset, hyponym)
        self.depth[new_synset] = min(
            [self.depth[hypernym] for hypernym in hypernyms]) + 1

    def del_node(self, synset):
        self.G.remove_node(synset)
        self.depth.pop(synset)

    def visualize(self, target=[]):
        G = self.G.to_undirected()
        pos = nx.spring_layout(G, seed=1)
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color=['r' if v in target else 'w' for v in G], alpha=0.5)
        nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels={v: v for v in G}, font_size=1, alpha=1.0)
        plt.savefig(f'{self.root}.png', dpi=1000, bbox_inches='tight', pad_inches=0)
