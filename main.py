import logging

import networkx as nx

from line.tasks import run
from utils import read_data

logging.basicConfig(
    format='%(process)d-%(levelname)s-%(message)s',
    level=logging.INFO)

if __name__ == '__main__':
    logging.info("Reading the graph data")
    edge_list = read_data("../graph-ml-project/data/out.munmun_twitter_social")
    graph = nx.DiGraph()

    logging.info("Constructed the graph")
    graph.add_weighted_edges_from(edge_list)
    run(graph=graph)
