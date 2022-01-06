import logging
import random

import networkx as nx

from line.model import Line
from netmf.model import NetMF
from utils import read_blog_catalog_edges, read_blog_catalog_labels, get_labels, node_classifier

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
    l = Line(train_graph=train_graph, batch_size=batch_size)
    l.run(epochs=n_iter)
    l.evaluate(test_graph)


def line_classification(graph, labels_dict, num_classes, n_iter=20, embedding_dim=128, batch_size=1024):
    """


    :param graph:
    :type graph:
    :param labels_dict:
    :type labels_dict: dict
    :param num_classes:
    :type num_classes:
    :param n_iter:
    :type n_iter:
    :param embedding_dim:
    :type embedding_dim:
    :param batch_size:
    :type batch_size:
    :return:
    :rtype:
    """
    line_class = Line(train_graph=graph, batch_size=batch_size, embedding_dim=embedding_dim)
    line_class.run(n_iter)
    embeddings = line_class.fetch_embedding_as_dict()
    embed_list = list()
    label_list = list()
    for node in embeddings.keys():
        embed_list.append(embeddings[node])
        label_list.append(labels_dict[node])
    print(node_classifier(embed_list, label_list))


# Function for Node Classification using NetMF
def netmf_node_classification(graph, labels, b, T, d=128, h=256, win_size="small"):
    X = NetMF(graph, win_size, b=b, T=T, d=d, iter=10, h=h)
    y = get_labels(graph.nodes(), labels)

    return node_classifier(X, y)


def main(file_loc="../graph-ml-project/data/out.munmun_twitter_social"):
    """

    :param file_loc:
    :type file_loc:
    :return:
    :rtype:
    """
    # Twitter data
    logging.info("Reading the graph data")
    # edge_list = read_twitter_edges(file_loc)
    # graph = nx.DiGraph()
    #
    # logging.info("Adding the edges")
    # graph.add_weighted_edges_from(edge_list)
    #
    # logging.info("Preparing train test split")
    # test_set = prepare_train_test(graph=graph)
    # test_graph = nx.DiGraph()
    # test_graph.add_weighted_edges_from(test_set)
    # train_graph = graph.copy()
    # train_graph.remove_edges_from(test_set)
    # del graph
    # logging.info("Constructed the graph")

    # BlogCatalog
    blog_edge_list = read_blog_catalog_edges("/content/soc-BlogCatalog-ASU.edges")
    blog_labels = read_blog_catalog_labels("/content/soc-BlogCatalog-ASU.node_labels")
    blog_graph = nx.Graph()
    blog_graph.add_weighted_edges_from(blog_edge_list)

    # PubMed
    pub_edge_list = read_pub_med_edges("/content/Pubmed-Diabetes.DIRECTED.cites.tab", 2)
    pub_labels = read_pub_med_labels("/content/Pubmed-Diabetes.NODE.paper.tab", 2)
    pub_graph = nx.Graph()
    pub_graph.add_weighted_edges_from(pub_edge_list)

    # NetMF NODE CLASSIFICATION
    # PubMed Large NetMF
    netmf_node_classification(pub_graph, pub_labels, b=1, T=10, win_size="large")
    # PubMed Small NetMF
    netmf_node_classification(pub_graph, pub_labels, b=1, T=1, win_size="small")

    # Blog Catalog Large NetMF
    netmf_node_classification(blog_graph, blog_labels, b=1, T=10, win_size="large")
    # Blog Catalog Small NetMF
    netmf_node_classification(blog_graph, blog_labels, b=1, T=1, win_size="small")
