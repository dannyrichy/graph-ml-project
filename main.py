import networkx as nx

from utils import read_data

if __name__ == '__main__':
    edge_list = read_data("../graph-ml-project/data/out.munmun_twitter_social")
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edge_list)

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