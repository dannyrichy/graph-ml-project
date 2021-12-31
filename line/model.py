import networkx as nx
import numpy as np

from utils import AliasTable


class LineBaseClass:
    """
    Line Base Class shizzzzzzzzzzzzzz
    """

    def __init__(self, graph):
        """

        :param graph: graph nodes
        :type graph: networkx.Graph
        """
        self.graph = graph

        # Preparing the edge distribution
        edge_distribution = np.array([
            attr['weight']
            for _, _, attr in self.graph.edges(data=True)
        ], dtype=np.float32)
        self.edge_distribution = edge_distribution / np.sum(edge_distribution)
        self.edge_alias_sampling = AliasTable(self.edge_distribution)

        # Preparing node distribution, using negative sampling to make life easier and using degree^3/4
        node_distribution = np.power(
            np.array([
                self.graph.degree(node, weight='weight')
                for node, _ in self.graph.nodes(data=True)
            ], dtype=np.float32),
            0.75)
        self.node_distribution = node_distribution / np.sum(node_distribution)
        self.node_alias_sampling = AliasTable(self.node_distribution)

        self.node_2_ix = dict()
        self.ix_2_node = dict()
        for index, (node, _) in enumerate(self.graph.nodes(data=True)):
            self.node_2_ix[node] = index
            self.ix_2_node[index] = node
        self.edges = [
            (self.node_2_ix[u], self.node_2_ix[v], _['weight'])
            for u, v, _ in self.graph.edges(data=True)
        ]

    def generate_batch_size(self, bs, num_negative_sample=10):
        """
        Generate batch
        :param bs:
        :type bs: int
        :param num_negative_sample:
        :type num_negative_sample: int
        :return:
        :rtype: dict
        """
        edge_batch = self.edge_alias_sampling.sampling(bs)
        v = {
            'v1': list(),
            'v2': list(),
            'weight': list(),
            'label': list()
        }

        for edge_ix in edge_batch:
            if (np.random.rand() > 0.5) & isinstance(self.graph, nx.Graph):
                v1, v2, weight = self.edges[edge_ix]
            else:
                v2, v1, weight = self.edges[edge_ix]
            v['v1'].append(v1)
            v['v2'].append(v2)
            v['weight'].append(weight)
            v['label'].append(1.0)

            for i in range(num_negative_sample):
                while True:
                    negative_node = self.node_alias_sampling.sampling()[0]
                    if not self.graph.has_edge(self.ix_2_node[negative_node],
                                               self.ix_2_node[v['v2'][-1]]):
                        v['v1'].append(v['v1'][-1])
                        v['v2'].append(negative_node)
                        v['weight'].append(0.0)
                        v['label'].append(-1.0)
                        break

        return v

    def embedding_mapping(self, embedding):
        return {
            node: embedding[self.node_2_ix[node]]
            for node, _ in self.graph.nodes(data=True)
        }
