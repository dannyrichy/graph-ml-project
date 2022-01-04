import networkx as nx

from utils import read_data, node2vec_random_walk
from gensim.models import Word2Vec

if __name__ == '__main__':
    edge_list = read_data("../graph-ml-project/data/out.munmun_twitter_social")
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_list)

    # build deepwalk corpus
    node2vec_paths = []
    walks_per_node = 1

    for node in graph:
        for _ in range(walks_per_node):
            node2vec_paths.append(node2vec_random_walk(graph, node, p=0.25, q=4))

    # initiate a word2vec model
    model = Word2Vec(window=2, sg=1, hs=0, negative=5,
                     vector_size=128, alpha=0.03, min_alpha=0.0001, seed=42)

    # build vocabulary
    model.build_vocab(node2vec_paths)

    # train
    model.train(node2vec_paths, total_examples=model.corpus_count,
                epochs=2, report_delay=1)

    print(model.wv['2'])