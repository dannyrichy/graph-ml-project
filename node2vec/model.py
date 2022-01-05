from gensim.models import Word2Vec

from utils import node2vec_random_walk, get_paths


def model(train_graph):
    node2vec_paths = get_paths(train_graph, node2vec_random_walk,
                               walks_per_node=1, walk_length=6, p=1, q=1)

    # initiate a word2vec model
    node2vec_model = Word2Vec(window=2, sg=1, hs=0, negative=5,
                              vector_size=128, alpha=0.03, min_alpha=0.0001, seed=42)

    # build vocabulary
    node2vec_model.build_vocab(node2vec_paths)

    # train
    node2vec_model.train(node2vec_paths, total_examples=node2vec_model.corpus_count,
                         epochs=2, report_delay=1)

    return node2vec_model
