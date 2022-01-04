import networkx as nx

from utils import read_data

if __name__ == '__main__':
    edge_list = read_data("../graph-ml-project/data/out.munmun_twitter_social")
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_list)

