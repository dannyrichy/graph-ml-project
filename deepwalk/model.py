from gensim.models import Word2Vec

from utils import deepwalk_random_walk, get_paths


def model(train_graph):
    deepwalk_paths = get_paths(train_graph, deepwalk_random_walk,
                               walks_per_node=1, walk_length=6, p=1, q=1)

    # initiate a deepwalk model
    deepwalk_model = Word2Vec(window=2, sg=1, hs=1, vector_size=128,
                              alpha=0.03, min_alpha=0.0001, seed=42)

    # build vocabulary
    deepwalk_model.build_vocab(deepwalk_paths)

    # train
    deepwalk_model.train(deepwalk_paths, total_examples=deepwalk_model.corpus_count,
                         epochs=2, report_delay=1)

    return deepwalk_model
