import logging
import random

import numpy as np

from line.utils import LABEL, V2, V1, WEIGHT
from utils import AliasTable


class LineBaseClass:
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

        self._prepare_aliases()

        self.node_2_ix = dict()
        self.ix_2_node = dict()
        for index, (node, _) in enumerate(self.graph.nodes(data=True)):
            self.node_2_ix[node] = index
            self.ix_2_node[index] = node

        self.edges = [
            (self.node_2_ix[u], self.node_2_ix[v])
            for u, v, _ in self.edges_nx
        ]

    def _prepare_aliases(self):
        logging.info("Preparing node distribution")
        # Preparing the edge distribution
        edge_distribution = np.array([
            attr[WEIGHT]
            for _, _, attr in self.graph.edges(data=True)
        ], dtype=np.float32)
        edge_distribution = edge_distribution / np.sum(edge_distribution)
        self.edge_alias_sampling = AliasTable(edge_distribution)

        # Preparing node distribution, using negative sampling to make life easier and using degree^3/4
        logging.info("Preparing edge distribution")
        node_distribution = np.power(
            np.array([
                self.graph.degree(node, weight=WEIGHT)
                for node, _ in self.nodes_nx
            ], dtype=np.float32),
            0.75)
        node_distribution = node_distribution / np.sum(node_distribution)
        self.node_alias_sampling = AliasTable(prob_dist=node_distribution)

    def batch_size_gen(self, batch_size):
        """
        Generator function to generate data samples
        :return:
        :rtype:
        """
        shuffle_indices = np.random.permutation(np.arange(self.num_edges))
        batches = [(i, min(i + batch_size, self.num_edges)) for i in range(0, self.num_edges, batch_size)]
        logging.debug("Batched indexes generated: {}".format(batches))
        while True:
            for batch_ixs in batches:
                batch_data = {
                    V1: list(),
                    V2: list(),
                    LABEL: list()
                }
                logging.debug("Preparing data sample for indexes: ({}, {})".format(*batch_ixs))
                for i in range(*batch_ixs):
                    if random.random() >= self.edge_alias_sampling.accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias_sampling.alias[shuffle_indices[i]]

                    v1, v2 = self.edges[shuffle_indices[i]]
                    batch_data[V1].append(v1)
                    batch_data[V2].append(v2)
                    batch_data[LABEL].append(1.0)

                    for _ in range(self.negative_ratio):
                        batch_data[V1].append(v1)
                        batch_data[V2].append(self.node_alias_sampling.alias_sample())
                        batch_data[LABEL].append(-1.0)

                yield ([np.array(batch_data[V1]), np.array(batch_data[V2])], [np.array(batch_data['label'])])

    def embedding_mapping(self, embedding):
        return {
            node: embedding[self.node_2_ix[node]]
            for node, _ in self.graph.nodes(data=True)
        }
