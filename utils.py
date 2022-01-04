"""
Common utility functions
"""
from operator import methodcaller
from typing import List

import numpy as np


# function to assign labels to nodes
def assign_labels(graph, labels_list, label_name='label'):
    for node, label in labels_list:
        graph.nodes[node][label_name] = label
    return graph


# read edges from Blog Catalog
def read_blog_catalog_edges(filename, header=0):
    with open(filename, 'r') as f:
        edge_list = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", ","), f.read().splitlines()[header:]))))
    return edge_list


# read labels from Blog Catalog
def read_blog_catalog_labels(filename, header=0):
    with open(filename, 'r') as f:
        labels = list(map(lambda x: (x[0], x[1]), list(map(methodcaller("split", ","), f.read().splitlines()[header:]))))
    return labels


# read edges from Twitter
def read_twitter_edges(filename, header=2):
    with open(filename, 'r') as f:
        list_edges = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", " "), f.read().splitlines()[header:]))))
    return list_edges



class AliasTable:
    def __init__(self, prob_dist):
        """
        Class to generate the alias table

        :param prob_dist: Probability distribution to use
        :type prob_dist: list

        :return: None
        :rtype: Nothing
        """
        self.prob = prob_dist
        self.num_pts = len(self.prob)
        self.accept = np.zeros(self.num_pts)
        self.alias = np.zeros(self.num_pts)
        self.create_alias_table()

    def create_alias_table(self):
        """
        Generates the alias and accept list
        :return: Nothing
        :rtype: None
        """
        small, large = list(), list()
        area_ratio_ = np.array(self.prob) * self.num_pts
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            self.accept[small_idx] = area_ratio_[small_idx]
            self.alias[small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            self.accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            self.accept[small_idx] = 1

    def alias_sample(self):
        """
        Sample from the generated list

        :return: index
        :rtype: int
        """
        i = int(np.random.random() * self.num_pts)
        return i if np.random.random() < self.accept[i] else self.alias[i]
