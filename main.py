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
    
    testNetMF("small")
    testNetMF("large")

    #BLOG CATALOG
    edge_list = read_BlogCatalog_edges("/content/soc-BlogCatalog-ASU.edges")
    labels = read_BlogCatalog_labels("/content/soc-BlogCatalog-ASU.node_labels")
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_list)
    graph = assign_labels(graph, labels)

    NetMF_Embedding = NetMF(graph, "large", b=5, T=10, d=2, iter=10, h=256)
    #NetMF_Embedding = NetMF(graph, "small", b=1, T=3, d=2, iter=10, h=256)

