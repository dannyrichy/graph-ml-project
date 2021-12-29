"""
Common utility function
"""
import math

import numpy as np


def create_alias_table(area_ratio):
    """

    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sampling(graph):
    """

    :param graph:
    :type graph: networkx.Graph
    :return:
    :rtype:
    """
    power = 0.75
    no_nodes = graph.number_of_nodes()
    no_edges = graph.number_of_edges()
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    node_degree = np.zeros(no_nodes)  # out degree

    for edge in graph.edges():
        node_degree[node2idx[edge[0]]] += graph[edge[0]][edge[1]].get('weight', 1.0)

    norm_prob = [float(math.pow(node_degree[j], power)) /
                 sum([math.pow(node_degree[i], power) for i in range(no_nodes)])
                 for j in range(no_nodes)]

    node_accept, node_alias = create_alias_table(norm_prob)

    # create sampling table for edge

    norm_prob = [graph[edge[0]][edge[1]].get('weight', 1.0) *
                 no_edges / sum([graph[edge[0]][edge[1]].get('weight', 1.0)
                                 for edge in graph.edges()])
                 for edge in graph.edges()]

    edge_accept, edge_alias = create_alias_table(norm_prob)
    return node_accept, node_alias, edge_accept, edge_alias, node2idx
