"""
Common utility functions
"""
from operator import methodcaller
#import tensorflow.experimental.numpy as np
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

#read edges from Blog Catalog
def read_BlogCatalog_edges(filename, header=0):
    with open(filename, 'r') as f:
        edge_list = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", ","), f.read().splitlines()[header:]))))
    return edge_list


#read labels from Blog Catalog
def read_BlogCatalog_labels(filename, header=0):
    with open(filename, 'r') as f:
        labels = list(map(lambda x: (x[0], x[1]), list(map(methodcaller("split", ","), f.read().splitlines()[header:]))))
    return labels


#function to assign labels to nodes 
def assign_labels(graph, labels_list, label_name='label'):
    for node,label in labels_list:
        graph.nodes[node][label_name] = label
    return graph


def read_data(filename, header=2):
    with open(filename, 'r') as f:
        list_edges = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", " "), f.read().splitlines()[header:]))))
    return list_edges

class AliasTable:
    def __init__(self, prob_dist):
        """
        Class to generate the alias table

        :param prob_dist: Probability distribution to use
        :type prob_dist: list

        :return: None
        :rtype: Nothing
        """
        self.prob = prob_dist
        self.num_pts = len(self.prob)
        self.accept = np.zeros(self.num_pts)
        self.alias = np.zeros(self.num_pts)
        self.create_alias_table()

    def create_alias_table(self):
        """
        Generates the alias and accept list
        :return: Nothing
        :rtype: None
        """
        small, large = list(), list()
        area_ratio_ = np.array(self.prob) * self.num_pts
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            self.accept[small_idx] = area_ratio_[small_idx]
            self.alias[small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            self.accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            self.accept[small_idx] = 1

    def alias_sample(self):
        """
        Sample from the generated list

        :return: index
        :rtype: int
        """
        i = int(np.random.random() * self.num_pts)
        return i if np.random.random() < self.accept[i] else self.alias[i]

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
    print("calculating D_inverse ...")
    L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
    d_rt_inv = sp.diags(d_rt ** -1)
    
    #Compute sum over P_r for r = 1 to T
    print("calculating sum over P_r ...")
    P = sp.identity(graph.number_of_nodes()) - L
    sum_P = np.zeros(P.shape)
    P_r = sp.identity(graph.number_of_nodes())
    for _ in range(T):
        P_r =  P_r.dot(P)
        sum_P = sum_P + P_r
    
    #Compute direct M
    print("computing direct M ...")
    M = (vol/(b*T)) *d_rt_inv.dot(d_rt_inv.dot(sum_P).T)
    
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
    print("performing Eigen-decomposition: D^{-1/2} A D^{-1/2} ...")
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
    print("computing approximate M ...")
    xx = sp.diags(np.sqrt(Lambda)).dot(D_invU.T).T
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
    print("performing truncatedSVD ...")
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
        print("Small NetMF:")
        m = compute_M_for_small_NetMF(graph, b, T)
    elif size == "large":
        print("Large NetMF")
        m = compute_M_for_large_NetMF(graph, b, T, h)
    else:
        print("size is either 'small' or 'large'")
        return

    NetMF_Embedding = create_embedding(m, d, iter)
    
    return NetMF_Embedding


#Test NetMF
def testNetMF(size):
    graph = nx.watts_strogatz_graph(1000, 10, 0.5)
    embedding = NetMF(graph, size, b=1, T=3, d=2, iter=10, h=256)

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 2
    assert type(embedding) == np.ndarray

    graph = nx.watts_strogatz_graph(1500, 10, 0.5)
    embedding = NetMF(graph, size, b=1, T=3, d=32, iter=10, h=256)

    assert embedding.shape[0] == graph.number_of_nodes()
    assert embedding.shape[1] == 32
    assert type(embedding) == np.ndarray