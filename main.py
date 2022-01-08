import logging

import networkx as nx
from stellargraph.data import EdgeSplitter

from deepwalk.model import Deepwalk
from line.model import Line
from netmf.model import NetMF
from node2vec.model import Node2Vec
from utils import *

logging.basicConfig(
    format='%(process)d-%(levelname)s-%(message)s',
    level=logging.INFO)


def prepare_train_test(graph, directed=False):
    """
    Train test split
    :param graph:
    :type graph: networkx.Graph
    :return:
    :rtype:
    """
    if directed:
        graph_copy = graph.copy()
        edge_splitter_test = EdgeSplitter(graph.to_undirected())
        _, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.5, keep_connected=True)
        for idx, edge in enumerate(examples_test[np.where(labels_test == 1)]):
            if not graph.has_edge(*tuple(edge)):
                examples_test[idx] = np.flip(edge)
        graph_copy.remove_edges_from(examples_test)
    else:
        edge_splitter_test = EdgeSplitter(graph)
        graph_copy, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.5, keep_connected=True)

    return graph_copy, examples_test, labels_test


def line_predictor(graph, n_iter=20, batch_size=1024, directed=False):
    """
    Link predictor for LINE model

    :param graph:
    :type graph: nx.Graph

    :param n_iter:
    :type n_iter: int

    :param batch_size:
    :type batch_size: int

    :return:
    :rtype: None
    """
    train_graph, test_set, labels = prepare_train_test(graph, directed)
    line_predict = Line(train_graph=train_graph, batch_size=batch_size)
    line_predict.run(epochs=n_iter)
    y_pred = line_predict.predict(test_set)
    print("0 %:", link_prediction(y_pred=y_pred, y_true=labels))

    test_set_50 = d_link_pred(graph, test_set, labels, p=0.5)
    y_pred = line_predict.predict(test_set_50)
    print("50 %:", link_prediction(y_pred=y_pred, y_true=labels))
    test_set_100 = d_link_pred(graph, test_set, labels, p=1.0)
    y_pred = line_predict.predict(test_set_100)
    print("100 %:", link_prediction(y_pred=y_pred, y_true=labels))


# Function for Link Prediction using NetMF
def netmf_link_prediction(graph, T, b=1, d=128, h=256, win_size="small"):
    train_graph, test_edges, labels = prepare_train_test(graph)

    embedding = NetMF(graph, win_size, b=b, T=T, d=d, iter=10, h=h)
    nodelist = list(graph.nodes())
    Z = {nodelist[i]: embedding[i] for i in range(len(graph.nodes()))}
    y_pred = np.array([np.dot(Z[edge[0]], Z[edge[1]]) for edge in test_edges])
    print(f"{win_size}NetMF Link Prediction Score: {link_prediction(y_pred, labels)}")
    return


# Function for Node Classification using LINE
def line_classification(graph, labels_dict, dataset, n_iter=20, embedding_dim=128, batch_size=1024):
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


# Function for Node Classification using Deepwalk
def deepwalk_node_classification(graph, labels, dataset):
    embeddings = Deepwalk(graph)

    X = []
    y = []
    for node in graph.nodes():
        X.append(embeddings[node])
        y.append(labels[node])

    results = node_classifier(X, y)
    store_node_classify_results(results, X, y, dataset, f"Deepwalk")
    return


# Function for Node Classification using Node2Vec
def node2vec_node_classification(graph, labels, dataset, p, q):
    embeddings = Node2Vec(graph, p, q)

    X = []
    y = []
    for node in graph.nodes():
        X.append(embeddings[node])
        y.append(labels[node])

    results = node_classifier(X, y)
    store_node_classify_results(results, X, y, dataset, f"Node2Vec")
    return


# Construct graph given dataset with path in Google drive
def construct_graph(dataset, directed=False):
    if dataset == "BlogCatalog" or dataset == "blogcatalog" or dataset == "Blog_Catalog" or dataset == "blog_catalog":
        blog_edge_list = read_edges("/content/drive/MyDrive/Datasets/soc-BlogCatalog-ASU.edges")
        blog_labels = read_labels("/content/drive/MyDrive/Datasets/soc-BlogCatalog-ASU.node_labels")
        blog_graph = nx.Graph()
        blog_graph.add_weighted_edges_from(blog_edge_list)
        print("Returning Blog Catalog graph and labels")
        return blog_graph, blog_labels

    elif dataset == "PubMed" or dataset == "pubmed" or dataset == "Pub_Med" or dataset == "pub_med":
        pub_edge_list = read_pub_med_edges("/content/drive/MyDrive/Datasets/Pubmed-Diabetes.DIRECTED.cites.tab")
        pub_labels = read_pub_med_labels("/content/drive/MyDrive/Datasets/Pubmed-Diabetes.NODE.paper.tab")
        pub_graph = nx.Graph()
        pub_graph.add_weighted_edges_from(pub_edge_list)
        print("Returning Pub Med graph and labels")
        return pub_graph, pub_labels

    elif dataset == "Cora" or dataset == "cora":
        cora_edge_list = read_edges("/content/drive/MyDrive/Datasets/out.subelj_cora_cora", " ", 2)
        cora_labels = read_cora_labels("/content/drive/MyDrive/Datasets/ent.subelj_cora_cora.class.name")
        if directed == True:
            cora_graph = nx.DiGraph()
            cora_graph.add_weighted_edges_from(cora_edge_list)
            print("Returning Cora directed graph")
            return cora_graph
        else:
            cora_graph = nx.Graph()
            cora_graph.add_weighted_edges_from(cora_edge_list)
            print("Returning Cora graph and labels")
            return cora_graph, cora_labels

    elif dataset == "Reddit" or dataset == "reddit":
        reddit_edge_list = read_reddit_edges("/content/drive/MyDrive/Datasets/reddit-G.pickle")
        reddit_labels = read_reddit_labels("/content/drive/MyDrive/Datasets/reddit-class_map.json")
        reddit_graph = nx.Graph()
        reddit_graph.add_weighted_edges_from(reddit_edge_list)
        print("Returning Reddit graph and labels")
        return reddit_graph, reddit_labels

    elif dataset == "Flickr" or dataset == "flickr":
        flickr_edge_list = read_edges("/content/drive/MyDrive/Datasets/soc-Flickr-ASU.edges")
        flickr_labels = read_labels("/content/drive/MyDrive/Datasets/soc-Flickr-ASU.node_labels")
        flickr_graph = nx.Graph()
        flickr_graph.add_weighted_edges_from(flickr_edge_list)
        print("Returning Flickr graph and labels")
        return flickr_graph, flickr_labels

    elif dataset == "Youtube" or dataset == "YouTube" or dataset == "youtube":
        youtube_edge_list = read_edges("/content/drive/MyDrive/Datasets/soc-YouTube-ASU.edges")
        youtube_labels = read_labels("/content/drive/MyDrive/Datasets/soc-YouTube-ASU.node_labels")
        youtube_graph = nx.Graph()
        youtube_graph.add_weighted_edges_from(youtube_edge_list)
        print("Returning Youtube graph and labels")
        return youtube_graph, youtube_labels

    elif dataset == "Facebook" or dataset == "FaceBook" or dataset == "facebook":
        facebook_edge_list = read_facebook_edges("/content/drive/MyDrive/Datasets/musae_facebook_edges.csv")
        facebook_labels = read_facebook_labels("/content/drive/MyDrive/Datasets/musae_facebook_target.csv")
        facebook_graph = nx.Graph()
        facebook_graph.add_weighted_edges_from(facebook_edge_list)
        print("Returning Facebook graph and labels")
        return facebook_graph, facebook_labels

    elif dataset == "Twitter" or dataset == "twitter":
        twitter_edge_list = read_edges("/content/drive/MyDrive/Datasets/out.munmun_twitter_social", " ", 2)
        twitter_graph = nx.DiGraph()
        twitter_graph.add_weighted_edges_from(twitter_edge_list)
        print("Returning Twitter directed graph")
        return twitter_graph

    elif dataset == "DBLP-Ci" or dataset == "dblp-ci":
        dblpci_edge_list = read_edges("/content/drive/MyDrive/Datasets/cit-DBLP.edges", " ", 2)
        dblpci_graph = nx.DiGraph()
        dblpci_graph.add_weighted_edges_from(dblpci_edge_list)
        print("Returning DBLP-Ci directed graph")
        return dblpci_graph

    elif dataset == "Epinion" or dataset == "epinion":
        epinion_edge_list = read_edges("/content/drive/MyDrive/Datasets/soc-Epinions1.mtx", " ", 2)
        epinion_graph = nx.DiGraph()
        epinion_graph.add_weighted_edges_from(epinion_edge_list)
        print("Returning Epinion directed graph")
        return epinion_graph

    elif dataset == "DBLP-Au" or dataset == "dblp-au":
        print("Dataset not yet available, try again!")
        return None

    else:
        print("Incorrect dataset name, try again!")
        return None


def main():
    """

    :param:
    :type:
    :return:
    :rtype:
    """

    # read and construct data
    dataset = "Cora"
    graph, labels = construct_graph(dataset)
    print(f"Directed:{nx.is_directed(graph)}, Edges:{graph.number_of_edges()}, Nodes: {graph.number_of_nodes()}, Labels: {len(labels)}")

    # node classification task
    line_classification(graph, labels, dataset)
    netmf_node_classification(graph, labels, dataset, T=1)
    netmf_node_classification(graph, labels, dataset, T=5, win_size="large")
    deepwalk_node_classification(graph, labels, dataset)
    node2vec_node_classification(graph, labels, dataset, p=0.25, q=4)

    # link prediction task
    netmf_link_prediction(graph, T=1)
    netmf_link_prediction(graph, T=5, win_size="large")

    # link prediction task
    netmf_link_prediction(graph, T=1)
    netmf_link_prediction(graph, T=5, win_size="large")

