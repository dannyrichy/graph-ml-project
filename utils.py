"""
Common utility function
"""
from operator import methodcaller

import numpy as np


def read_data(filename, header=2):
    with open(filename, 'r') as f:
        list_edges = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", " "), f.read().splitlines()[header:]))))
    return list_edges


class AliasTable:
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
