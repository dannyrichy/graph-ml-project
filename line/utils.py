import logging
import random

import numpy as np

from line import WEIGHT
from utils import AliasTable


def shuffle_det(*args):
    """
    To shuffle lists without changing the relation
    :param args: list
    :type args: list
    :return: list of shuffled items
    :rtype: list
    """
    tmp = list(zip(*args))
    random.shuffle(tmp)
    return list(zip(*tmp))


class GraphHelper:
    """
    Line Base Class shizzzzzzzzzzzzzz
    """
    edge_alias_sampling, node_alias_sampling = None, None

    def __init__(self, graph, negative_ratio):
        """

        :param graph: graph nodes
        :type graph: networkx.Graph
        """
        logging.debug("Initialising base line class with negative ratio : {}".format(negative_ratio))
        self.graph = graph
        self.negative_ratio = negative_ratio

        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        logging.debug("Number of nodes: {}, Number of edges: {}".format(self.num_nodes, self.num_edges))

        self.nodes_nx = self.graph.nodes(data=True)
        self.edges_nx = self.graph.edges(data=True)

        self.node_2_ix = dict()
        self.ix_2_node = dict()
        for index, (node, _) in enumerate(self.nodes_nx):
            self.node_2_ix[node] = index
            self.ix_2_node[index] = node

        self.edges = [
            (self.node_2_ix[u], self.node_2_ix[v])
            for u, v, _ in self.edges_nx
        ]
        self._prepare_aliases()

    def _prepare_aliases(self):
        logging.info("Preparing node distribution")
        # Preparing the edge distribution
        edge_distribution = np.array([
            attr[WEIGHT]
            for _, _, attr in self.edges_nx
        ], dtype=np.float32)
        edge_distribution = edge_distribution * (self.num_edges / np.sum(edge_distribution))
        self.edge_alias_sampling = AliasTable(edge_distribution)

        # Preparing node distribution, using negative sampling to make life easier and using degree^3/4
        logging.info("Preparing edge distribution")
        node_degree = np.zeros(self.num_nodes)
        for edge in self.graph.edges():
            node_degree[self.node_2_ix[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)
        node_distribution = np.power(
            np.array([
                node_degree[j]
                for j in range(self.num_nodes)
            ], dtype=np.float32),
            0.75)
        node_distribution = node_distribution / np.sum(node_distribution)
        self.node_alias_sampling = AliasTable(prob_dist=node_distribution)
