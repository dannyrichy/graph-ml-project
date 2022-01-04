import networkx as nx
import numpy as np

from sklearn.decomposition import TruncatedSVD


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



