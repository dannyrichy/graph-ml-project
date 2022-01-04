import logging
import random

import networkx as nx

from line.tasks import LinkPredict
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
    return random.sample([(u, v, weight['weight']) for u, v, weight in graph.edges(data=True)], int(0.25 * graph.number_of_edges()))


def main(file_loc="../graph-ml-project/data/out.munmun_twitter_social", n_iter=20, batch_size=1024):
    logging.info("Reading the graph data")
    edge_list = read_data(file_loc)
    graph = nx.DiGraph()

    logging.info("Adding the edges")
    graph.add_weighted_edges_from(edge_list)

    logging.info("Preparing train test split")
    test_set = prepare_train_test(graph=graph)
    test_graph = nx.DiGraph()
    test_graph.add_weighted_edges_from(test_set)
    train_graph = graph.copy()
    train_graph.remove_edges_from(test_set)
    del graph
    logging.info("Constructed the graph")

    l = LinkPredict(train_graph=train_graph, batch_size=batch_size)
    l.run(epochs=n_iter)
    l.evaluate(test_graph)