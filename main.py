import logging
import random

import networkx as nx
from stellargraph.data import EdgeSplitter

from line.model import Line
from netmf.model import NetMF
from utils import *


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
    edge_splitter_test = EdgeSplitter(graph)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.5, keep_connected=True)
    return graph_test, examples_test, labels_test


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


def line_classification(graph, labels_dict, dataset, n_iter=20, embedding_dim=128, batch_size=1024):
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
    results = node_classifier(embed_list, label_list)
    store_node_classify_results(results, embed_list, label_list, dataset, "LINE")
    return


# Function for Node Classification using NetMF
def netmf_node_classification(graph, labels, dataset, T, b=1, d=128, h=256, win_size="small"):
    X = NetMF(graph, win_size, b=b, T=T, d=d, iter=10, h=h)
    y = get_labels(graph.nodes(), labels)
    results = node_classifier(X, y)
    store_node_classify_results(results, X, y, dataset, f"{win_size}NetMF")
    return 


def main():
    """

    :param:
    :type:
    :return:
    :rtype:
    """

    ##### READ and CONSTRUCT all DATATSETS #####
    
    # BlogCatalog
    blog_edge_list = read_soc_edges("/content/drive/MyDrive/Datasets/soc-BlogCatalog-ASU.edges")
    blog_labels = read_soc_labels("/content/drive/MyDrive/Datasets/soc-BlogCatalog-ASU.node_labels")
    blog_graph = nx.Graph()
    blog_graph.add_weighted_edges_from(blog_edge_list)

    # PubMed
    pub_edge_list = read_pub_med_edges("/content/drive/MyDrive/Datasets/Pubmed-Diabetes.DIRECTED.cites.tab")
    pub_labels = read_pub_med_labels("/content/drive/MyDrive/Datasets/Pubmed-Diabetes.NODE.paper.tab")
    pub_graph = nx.Graph()
    pub_graph.add_weighted_edges_from(pub_edge_list)

    # Flickr
    flickr_edge_list = read_soc_edges("/content/drive/MyDrive/Datasets/soc-Flickr-ASU.edges")
    flickr_labels = read_soc_labels("/content/drive/MyDrive/Datasets/soc-Flickr-ASU.node_labels")
    flickr_graph = nx.Graph()
    flickr_graph.add_weighted_edges_from(flickr_edge_list)

    # Youtube
    youtube_edge_list = read_soc_edges("/content/drive/MyDrive/Datasets/soc-YouTube-ASU.edges")
    youtube_labels = read_soc_labels("/content/drive/MyDrive/Datasets/soc-YouTube-ASU.node_labels")
    youtube_graph = nx.Graph()
    youtube_graph.add_weighted_edges_from(youtube_edge_list)

    # Cora
    cora_edge_list = read_cora_edges("/content/drive/MyDrive/Datasets/out.subelj_cora_cora")
    cora_labels = read_cora_labels("/content/drive/MyDrive/Datasets/ent.subelj_cora_cora.class.name")
    cora_graph = nx.Graph()
    cora_graph.add_weighted_edges_from(cora_edge_list)

    # Reddit
    reddit_adjlist = open("/content/drive/MyDrive/Datasets/reddit-adjlist.txt", 'rb')
    reddit_graph = nx.read_adjlist(reddit_adjlist, comments='#')
    reddit_labels = read_reddit_labels("/content/drive/MyDrive/Datasets/reddit-class_map.json")

    # Facebook
    facebook_edge_list = read_facebook_edges("/content/drive/MyDrive/Datasets/musae_facebook_edges.csv")
    facebook_labels = read_facebook_labels("/content/drive/MyDrive/Datasets/musae_facebook_target.csv")
    facebook_graph = nx.Graph()
    facebook_graph.add_weighted_edges_from(facebook_edge_list)
    

    ##### NODE CLASSIFICATION #####
    
    # BlogCatalog
    line_classification(blog_graph, blog_labels, "BlogCatalog")
    netmf_node_classification(blog_graph, blog_labels, "BlogCatalog", T=1)
    netmf_node_classification(blog_graph, blog_labels, "BlogCatalog", T=5, win_size="large")    
    
    
    # PubMed
    line_classification(pub_graph, pub_labels, "PubMed")
    netmf_node_classification(pub_graph, pub_labels, "PubMed", T=1)
    netmf_node_classification(pub_graph, pub_labels, "PubMed", T=5, win_size="large")   
    
    
    # Flickr
    line_classification(flickr_graph, flickr_labels, "Flickr")
    netmf_node_classification(flickr_graph, flickr_labels, "Flickr", T=1, h=16389)
    netmf_node_classification(flickr_graph, flickr_labels, "Flickr", T=5, win_size="large", h=16389)
    
    
    # Youtube
    line_classification(youtube_graph, youtube_labels, "Youtube")
    netmf_node_classification(youtube_raph, youtube_labels, "Youtube", T=1)
    netmf_node_classification(youtube_graph, youtube_labels, "Youtube", T=5, win_size="large")    
    
    
    # Cora
    line_classification(cora_graph, cora_labels, "Cora")
    netmf_node_classification(cora_graph, cora_labels, "Cora", T=1)
    netmf_node_classification(cora_graph, cora_labels, "Cora", T=5, win_size="large")
    
    
    # Reddit
    line_classification(reddit_graph, reddit_abels, "Reddit")
    netmf_node_classification(reddit_graph, reddit_labels, "Reddit", T=1)
    netmf_node_classification(reddit_graph, reddit_labels, "Reddit", T=5, win_size="large")   
    
    
    # Facebook
    line_classification(facebook_graph, facebook_labels, "Facebook")
    netmf_node_classification(facebook_graph, facebook_labels, "Facebook", T=1)
    netmf_node_classification(facebook_graph, facebook_labels, "Facebook", T=5, win_size="large")
    
    
    
