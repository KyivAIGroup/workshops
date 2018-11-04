from collections import defaultdict
from functools import total_ordering
from itertools import product, repeat
from pprint import pprint

import numpy as np


@total_ordering
class Cluster:
    def __init__(self, code):
        data_clustered = defaultdict(list)
        for assignment, cluster_id in enumerate(code):
            same = repeat(unique[assignment], times=counts[assignment])
            data_clustered[cluster_id].extend(same)
        entropy = 0
        for elements in data_clustered.values():
            elements.sort()
            _, appearances = np.unique(elements, return_counts=True)
            proba_item = appearances / sum(appearances)
            entropy += sum(proba_item * np.log2(1. / proba_item))
        # convert lists to tuples
        data_clustered = tuple(sorted(map(tuple, data_clustered.values())))
        std = sum(map(np.std, data_clustered))

        self.clustered = data_clustered
        self.entropy = entropy
        self.std = std

    def __eq__(self, other):
        return self.clustered == self.clustered

    def __hash__(self):
        return hash(self.clustered)

    def __lt__(self, other):
        return (self.entropy, -self.std) < (other.entropy, -other.std)

    def __repr__(self):
        return f"Cluster({self.clustered}, #clusters={len(self.clustered)}, " \
               f"entropy={self.entropy:.3f}, std={self.std:.3f})"


def make_clusters_brute_force(n_clusters: int):
    codes = product(range(n_clusters), repeat=len(unique))
    codes = filter(lambda code: len(set(code)) == n_clusters, codes)
    clusters = [Cluster(code=code) for code in codes]
    return clusters


data = [9, 10, 10, 2, 1]

unique, counts = np.unique(data, return_counts=True)

clusters = set()

for n_clusters in range(1, len(unique) + 1):
    clusters.update(make_clusters_brute_force(n_clusters=n_clusters))

clusters = sorted(clusters, reverse=True)
pprint(clusters)
