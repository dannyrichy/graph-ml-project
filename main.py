import logging
import random

import networkx as nx

from line.tasks import Line1
from utils import read_data

logging.basicConfig(
    format='%(process)d-%(levelname)s-%(message)s',
    level=logging.INFO)


def prepare_train_test(graph):
    """
    Train test split
    :param graph:
    :type graph: networkx.Graph
    :return:
    :rtype:
    """
    edge_subset = random.sample(graph.edges(), int(0.25 * graph.number_of_edges()))
    edge_test_subset = list(graph.edges())
    for edge in edge_subset:
        edge_test_subset.remove(edge)
    train_g = graph.copy()
    train_g = train_g.remove_edges_from(edge_subset)
    test_g = graph.remove_edges_from(edge_test_subset)
    return train_g, test_g


if __name__ == '__main__':
    logging.info("Reading the graph data")
    edge_list = read_data("../graph-ml-project/data/out.munmun_twitter_social")

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edge_list)
    train_graph, test_graph = prepare_train_test(graph=graph)
    logging.info("Constructed the graph")

    l = Line1(graph=graph)
    l.run(no_iter=100)

