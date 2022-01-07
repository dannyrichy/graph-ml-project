from gensim.models import Word2Vec

from deepwalk.utils import deepwalk_random_walk, get_paths


def Deepwalk(train_graph):
    deepwalk_paths = get_paths(train_graph, deepwalk_random_walk,
                               walks_per_node=20, walk_length=10, p=1, q=1)

    # initiate a deepwalk model
    deepwalk_model = Word2Vec(vector_size=128, window=5, min_count=1, workers=4, sg=1, hs=1,
                              alpha=0.03, min_alpha=0.0001, seed=42, sample=0)

    # build vocabulary
    deepwalk_model.build_vocab(deepwalk_paths)

    # train
    deepwalk_model.train(deepwalk_paths, total_examples=deepwalk_model.corpus_count,
                         epochs=10, report_delay=1)

    return deepwalk_model.wv
