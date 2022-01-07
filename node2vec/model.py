from gensim.models import Word2Vec

from node2vec.utils import node2vec_random_walk, get_paths


def Node2Vec(train_graph, p, q):
    node2vec_paths = get_paths(train_graph, node2vec_random_walk,
                               walks_per_node=10, walk_length=5, p=p, q=q)

    # initiate a word2vec model
    node2vec_model = Word2Vec(vector_size=128, window=2, min_count=1, workers=4, sg=1, hs=0, negative=5,
                              alpha=0.03, min_alpha=0.0001, seed=42, sample=0)

    # build vocabulary
    node2vec_model.build_vocab(node2vec_paths)

    # train
    node2vec_model.train(node2vec_paths, total_examples=node2vec_model.corpus_count,
                         epochs=20, report_delay=1)

    return node2vec_model.wv
