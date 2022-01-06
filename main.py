import logging
import random

import networkx as nx

from line.model import Line
from netmf.model import NetMF
from utils import read_soc_edges, read_soc_labels, get_labels, node_classifier, read_pub_med_edges, read_pub_med_labels, read_cora_edges, read_cora_labels

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


def line_classification(graph, labels_dict, n_iter=20, embedding_dim=128, batch_size=1024):
    """


    :param graph:
    :type graph:
    :param labels_dict:
    :type labels_dict: dict

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


def main():
    """

    :param:
    :type:
    :return:
    :rtype:
    """

    # BlogCatalog
    blog_edge_list = read_soc_edges("/content/soc-BlogCatalog-ASU.edges")
    blog_labels = read_soc_labels("/content/soc-BlogCatalog-ASU.node_labels")
    blog_graph = nx.Graph()
    blog_graph.add_weighted_edges_from(blog_edge_list)

    # PubMed
    pub_edge_list = read_pub_med_edges("/content/Pubmed-Diabetes.DIRECTED.cites.tab")
    pub_labels = read_pub_med_labels("/content/Pubmed-Diabetes.NODE.paper.tab")
    pub_graph = nx.Graph()
    pub_graph.add_weighted_edges_from(pub_edge_list)

    # Flickr
    flickr_edge_list = read_soc_edges("/content/drive/MyDrive/soc-Flickr-ASU.edges")
    flickr_labels = read_soc_labels("/content/drive/MyDrive/soc-Flickr-ASU.node_labels")
    flickr_graph = nx.Graph()
    flickr_graph.add_weighted_edges_from(flickr_edge_list)

    # Youtube
    youtube_edge_list = read_soc_edges("/content/drive/MyDrive/soc-YouTube-ASU.edges")
    youtube_labels = read_soc_labels("/content/drive/MyDrive/soc-YouTube-ASU.node_labels")
    youtube_graph = nx.Graph()
    youtube_graph.add_weighted_edges_from(flickr_edge_list)
    
    # Cora
    cora_edge_list = read_cora_edges("/content/drive/MyDrive/out.subelj_cora_cora")
    cora_labels = read_cora_labels("/content/drive/MyDrive/ent.subelj_cora_cora.class.name")
    cora_graph = nx.Graph()
    cora_graph.add_weighted_edges_from(cora_edge_list)
    

    # NetMF NODE CLASSIFICATION
    # PubMed Large NetMF
    netmf_node_classification(pub_graph, pub_labels, b=1, T=10, win_size="large")
    # PubMed Small NetMF
    netmf_node_classification(pub_graph, pub_labels, b=1, T=1, win_size="small")

    # Blog Catalog Large NetMF
    netmf_node_classification(blog_graph, blog_labels, b=1, T=10, win_size="large")
    # Blog Catalog Small NetMF
    netmf_node_classification(blog_graph, blog_labels, b=1, T=1, win_size="small")
    # Blog Catalog LINE
    line_classification(blog_graph, blog_labels)

    # Youtube Small NetMF
    netmf_node_classification(youtube_graph, youtube_labels, b=1, T=1)
    # Youtube Line
    line_classification(youtube_graph, youtube_labels)

    # Flickr Small NetMF
    netmf_node_classification(flickr_graph, flickr_labels, b=1, T=1, h=16389)
    # Flickr Line
    line_classification(flickr_graph, flickr_labels)
    
    # Cora Line
    line_classification(cora_graph, cora_labels)
    
    # Cora Small NetMF
    netmf_node_classification(cora_graph, cora_labels, b=1, T=1)
