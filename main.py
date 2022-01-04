import logging
import random

import networkx as nx

from line.model import LinkPredict
from netmf.model import NetMF
from utils import read_twitter_edges, read_blog_catalog_edges, read_blog_catalog_labels, assign_labels

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


def line_predictor(train_graph, test_graph, n_iter=20, batch_size=1024):
    """
    Link predictor for LINE model

    :param train_graph:
    :type train_graph: nx.Graph

    :param test_graph:
    :type test_graph: nx.Graph

    :param n_iter:
    :type n_iter: int

    :param batch_size:
    :type batch_size: int

    :return:
    :rtype: None
    """
    l = LinkPredict(train_graph=train_graph, batch_size=batch_size)
    l.run(epochs=n_iter)
    l.evaluate(test_graph)


def netmf(train_graph, test_graph, b, T, win_size="small"):
    NetMF_Embedding = NetMF(train_graph, win_size, b=b, T=T, d=2, iter=10, h=256)
    # NetMF_Embedding = NetMF(graph, "small", b=1, T=3, d=2, iter=10, h=256)


def main(file_loc="../graph-ml-project/data/out.munmun_twitter_social"):
    # Twitter data
    logging.info("Reading the graph data")
    edge_list = read_twitter_edges(file_loc)
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

    # BlogCatalog
    edge_list = read_blog_catalog_edges("/content/soc-BlogCatalog-ASU.edges")
    labels = read_blog_catalog_labels("/content/soc-BlogCatalog-ASU.node_labels")
    blog_catalog_graph = nx.Graph()
    blog_catalog_graph.add_weighted_edges_from(edge_list)
    blog_catalog_graph = assign_labels(blog_catalog_graph, labels)


def node2vec_link_predictor():
    pass
