"""
Common utility function
"""
import logging
from operator import methodcaller

import numpy as np


def read_data(filename, header=2):
    with open(filename, 'r') as f:
        list_edges = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", " "), f.read().splitlines()[header:]))))
    return list_edges


class AliasTableOld:
    """
    Class to perform alias table method for a given discrete distribution
    """
    smaller = list()
    larger = list()

    def __init__(self, prob):
        """
        Initialise with probability distribution
        :param prob: probability distribution
        :type prob: list
        """
        self.prob = prob
        self.num_points = len(prob)
        self.q = np.zeros(self.num_points)
        self.j = np.zeros(self.num_points)

        # Storing the indexes in the corresponding lists
        for ix, prob in enumerate(self.prob):
            self.q[ix] = prob * self.num_points
            if prob < 1 / self.num_points:
                self.smaller.append(ix)
            else:
                self.larger.append(ix)

        self._prepare()

    def _prepare(self):
        """
        Internal method to prepare the alias table method

        :return: Nothing
        :rtype: None
        """
        while len(self.smaller) > 0 and len(self.larger) > 0:
            # Popping the indexes from each of the list
            ix1 = self.smaller.pop()
            ix2 = self.larger.pop()

            # Storing the indexes corresponding to larger values to places where the
            self.j[ix1] = ix2
            self.q[ix2] -= 1 - self.q[ix1]

            if self.q[ix2] < 1.0:
                self.smaller.append(ix2)
            else:
                self.larger.append(ix2)

    def sampling(self, n=1):
        """
        Samples n points from the alias table method

        :param n: Number of  points to return
        :type n: int

        :return: list of sampled points
        :rtype: list
        """
        x = np.random.rand(n)
        ixs = np.floor(self.num_points * x).astype(np.int32)
        y = self.num_points * x - ixs
        return [ixs[k] if y[k] < self.q[ixs[k]] else self.j[ixs[k]] for k in range(n)]


class AliasTable:
    def __init__(self, prob_dist):
        """

        :param prob_dist:
        :type prob_dist: list
        :return:
        :rtype:
        """
        logging.info("Creating Alias Table")
        self.prob = prob_dist
        self.len = len(self.prob)
        self.accept = [0] * self.len
        self.alias = [0] * self.len
        self.create_alias_table()

    def create_alias_table(self):
        small, large = [], []
        area_ratio_ = np.array(self.prob) * self.len
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            self.accept[small_idx] = area_ratio_[small_idx]
            self.alias[small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                     (1 - area_ratio_[small_idx])
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
        :return: sample index
        """
        N = len(self.accept)
        i = int(np.random.random() * N)
        r = np.random.random()
        if r < self.accept[i]:
            return i
        else:
            return self.alias[i]
