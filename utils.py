"""
Common utility function
"""
import math
from operator import methodcaller

import tensorflow.experimental.numpy as np

import networkx as nx
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

def read_BlogCatalog(filename, header=0):
    with open(filename, 'r') as f:
        list_edges = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", ","), f.read().splitlines()[header:]))))
    return list_edges


def read_data(filename, header=2):
    with open(filename, 'r') as f:
        list_edges = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", " "), f.read().splitlines()[header:]))))
    return list_edges


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


# Compute log M' for small NetMF
def compute_M_for_small_NetMF(graph, b, T):
    """
  
    arguments:
    graph - main input
    b - no. of negative samples
    T - window

    returns:
    m - exact log M'
    """
    A = nx.adjacency_matrix(graph)
    vol = float(A.sum())

    #D_inverse = np.diag(np.array([val for (node, val) in graph.degree()]))
    #P = D_inverse @ A

    #Calculating D_inverse and 
    print("Calculating D_inverse")
    L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
    d_rt_inv = sp.diags(d_rt ** -1)
    
    #Compute sum over P_r for r = 1 to T
    print("Calculating sum over P_r")
    P = sp.identity(graph.number_of_nodes()) - L
    sum_P = np.zeros(P.shape)
    P_r = sp.identity(graph.number_of_nodes())
    for _ in range(T):
        P_r =  P_r.dot(P)
        sum_P = sum_P + P_r
    
    #Compute direct M
    print("Computing direct M")
    M = (vol/(b*T)) *d_rt_inv.dot(d_rt_inv.dot(sum_P).T).todense()
    
    #Compute log M'
    M[M<=1] = 1
    m = np.log(M)
    
    return m


# Compute log M' for large NetMF
def compute_M_for_large_NetMF(graph, b, T, h):
    """
    arguments:
    graph - main input
    b - no. of negative samples
    T - window
    h - rank of eigen decomposition
    
    returns:
    m - approximate log M'
    """
    A = nx.adjacency_matrix(graph)
    vol = float(A.sum())
    
    #Eigen-decomposition: D^{-1/2} A D^{-1/2}
    print("Performing Eigen-decomposition: D^{-1/2} A D^{-1/2}")
    L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
    D = sp.identity(graph.number_of_nodes()) - L
    Lambda, eigenvec = sp.linalg.eigsh(D, h)
    D_inv = sp.diags(d_rt ** -1)
    D_invU = D_inv.dot(eigenvec)
    
    #Apply summation over window: 1/T sum(Lambda^r) for r=1 to T
    for i in range(len(Lambda)):
        s = Lambda[i]
        Lambda[i] =  1.0 if s>=0 else (s*(1-s**T))/((1-s)*T)
    Lambda = np.maximum(Lambda, 0)
    
    #Approximate M 
    print("Computing approximate M")
    xx = sp.diags(np.dot(np.sqrt(Lambda),D_invU.T)).T
    M = (vol/b) * np.dot(xx, xx.T)
    
    #Compute log M'
    M[M<=1] = 1
    m = np.log(M)
    
    return m


# Create Embedding using SVD for NetMF
def create_embedding(m, d, iter):
    """
    arguments:
    m - log M'
    d - embedding space
    iter - iterations of svd
    
    returns:
    embedding - embedding of graph in d-space using svd
    """
    svd = TruncatedSVD(n_components=d, n_iter=iter, random_state=420)
    svd.fit(m)
    embedding = svd.transform(m)
    
    return embedding


# NetMF function
def NetMF(graph, size, b, T, d, iter, h):
    """
    arguments:
    graph - main input
    b - no. of negative samples
    T - window
    d - embedding 
    h - rank of eigen decomposition
    iter - iterations of svd
    
    returns:
    NetMF_Embedding - embedding of graph in d-space
    """
    if size == "small":
        print("Small NetMF")
        m = compute_M_for_small_NetMF(graph, b, T)
    elif size == "large":
        print("Large NetMF")
        m = compute_M_for_large_NetMF(graph, b, T, h)
    else:
        print("size is either 'small' or 'large'")
        return

    print("Creating Embedding")
    NetMF_Embedding = create_embedding(m, d, iter)
    
    return NetMF_Embedding