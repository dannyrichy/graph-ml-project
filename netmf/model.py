import scipy.sparse as sp

from netmf.utils import create_embedding


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

    # D_inverse = np.diag(np.array([val for (node, val) in graph.degree()]))
    # P = D_inverse @ A

    # Calculating D_inverse and
    print("calculating D_inverse ...")
    L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
    d_rt_inv = sp.diags(d_rt ** -1)

    # Compute sum over P_r for r = 1 to T
    print("calculating sum over P_r ...")
    P = sp.identity(graph.number_of_nodes()) - L
    sum_P = np.zeros(P.shape)
    P_r = sp.identity(graph.number_of_nodes())
    for _ in range(T):
        P_r = P_r.dot(P)
        sum_P = sum_P + P_r

    # Compute direct M
    print("computing direct M ...")
    M = (vol / (b * T)) * d_rt_inv.dot(d_rt_inv.dot(sum_P).T)

    # Compute log M'
    M[M <= 1] = 1
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

    # Eigen-decomposition: D^{-1/2} A D^{-1/2}
    print("performing Eigen-decomposition: D^{-1/2} A D^{-1/2} ...")
    L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
    D = sp.identity(graph.number_of_nodes()) - L
    Lambda, eigenvec = sp.linalg.eigsh(D, h)
    D_inv = sp.diags(d_rt ** -1)
    D_invU = D_inv.dot(eigenvec)

    # Apply summation over window: 1/T sum(Lambda^r) for r=1 to T
    for i in range(len(Lambda)):
        s = Lambda[i]
        Lambda[i] = 1.0 if s >= 0 else (s * (1 - s ** T)) / ((1 - s) * T)
    Lambda = np.maximum(Lambda, 0)

    # Approximate M
    print("computing approximate M ...")
    xx = sp.diags(np.sqrt(Lambda)).dot(D_invU.T).T
    M = (vol / b) * np.dot(xx, xx.T)

    # Compute log M'
    M[M <= 1] = 1
    m = np.log(M)

    return m


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


# Test NetMF
# def testNetMF(win_size):
#     graph = nx.watts_strogatz_graph(1000, 10, 0.5)
#     embedding = NetMF(graph, win_size, b=1, T=3, d=2, iter=10, h=256)
#
#     assert embedding.shape[0] == graph.number_of_nodes()
#     assert embedding.shape[1] == 2
#     assert type(embedding) == np.ndarray
#
#     graph = nx.watts_strogatz_graph(1500, 10, 0.5)
#     embedding = NetMF(graph, win_size, b=1, T=3, d=32, iter=10, h=256)
#
#     assert embedding.shape[0] == graph.number_of_nodes()
#     assert embedding.shape[1] == 32
#     assert type(embedding) == np.ndarray
