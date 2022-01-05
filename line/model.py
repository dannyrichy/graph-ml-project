import logging
import random
from itertools import islice

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as k

from line import V1, V2, LABEL
from line.utils import shuffle_det, GraphHelper, MyLRSchedule


def loss_fun(y_true, y_predicted):
    return -k.mean(k.log(k.sigmoid(y_true * y_predicted)))


class Line(GraphHelper):
    def __init__(self, train_graph, batch_size=128, negative_ratio=5, embedding_dim=128):
        """
        Class for link prediction

        :param train_graph: Graph to train on
        :type train_graph: networkx.Graph

        :param batch_size: Batch size
        :type batch_size: int

        :param negative_ratio: Number of negative samples to construct
        :type negative_ratio: int

        :param embedding_dim: Embedding dimension
        :type embedding_dim: int
        """
        logging.info("Initialising the base class")
        super().__init__(train_graph, negative_ratio)

        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Defining the input block
        logging.info("Creating the model")
        self.model, self.embed = self._create_model()

    def _create_model(self):
        """
        Creates tensorflow model

        :return: model and the embedding
        :rtype: (tensorflow.keras.models.Model, tensorflow.keras.layers.Embedding)
        """
        node1 = layers.Input(shape=(1,))
        node2 = layers.Input(shape=(1,))

        embed = layers.Embedding(self.num_nodes, self.embedding_dim, name='first_emb')

        u1 = embed(node1)
        u2 = embed(node2)

        out = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=-1, keepdims=False), name='LINE-1')([u1, u2])

        return Model(inputs=[node1, node2], outputs=[out]), embed

    def compile_model(self):
        logging.info("Compiling a model with Adam optimizer")
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(initial_learning_rate=0.025, max_t=10)),
                           loss=loss_fun,
                           metrics=[tf.keras.metrics.Accuracy()])

    def run(self, epochs, ):
        """
        Fit the model

        :param epochs: Number of epochs
        :type epochs: int

        :return: Nothing
        :rtype: None
        """
        self.compile_model()

        logging.info("Fitting the model with batch size:{}, epochs:{}".format(self.batch_size, epochs))
        batch_gen = self._batch_size_gen(self.batch_size)
        self.model.fit(batch_gen, epochs=epochs, steps_per_epoch=((self.num_edges * (1 + self.negative_ratio) - 1) // self.batch_size + 1))

    def _batch_size_gen(self, batch_size):
        """
        Generator function to generate data samples
        :return:
        :rtype:
        """
        shuffle_indices = np.random.permutation(np.arange(self.num_edges))
        batches = [(i, min(i + batch_size, self.num_edges)) for i in range(0, self.num_edges, batch_size)]
        logging.debug("Batched indexes generated: {}".format(batches))
        while True:
            for batch_ixs in batches:
                batch_data = {
                    V1: list(),
                    V2: list(),
                    LABEL: list()
                }
                logging.debug("Preparing data sample for indexes: ({}, {})".format(*batch_ixs))
                for i in range(*batch_ixs):
                    if random.random() >= self.edge_alias_sampling.accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias_sampling.alias[shuffle_indices[i]]

                    v1, v2 = self.edges[shuffle_indices[i]]
                    batch_data[V1].append(v1)
                    batch_data[V2].append(v2)
                    batch_data[LABEL].append(1.0)

                    for _ in range(self.negative_ratio):
                        batch_data[V1].append(v1)
                        batch_data[V2].append(self.node_alias_sampling.alias_sample())
                        batch_data[LABEL].append(-1.0)

                shuffled_data = {
                    key: value
                    for key, value in zip([V1, V2, LABEL], shuffle_det(batch_data[V1], batch_data[V2], batch_data[LABEL]))
                }
                mini_batches = [
                    (i, min(i + batch_size, 6 * np.diff(batch_ixs)[0]))
                    for i in range(0, 6 * np.diff(batch_ixs)[0], batch_size)
                ]
                for mini_batch_ixs in mini_batches:
                    yield ([np.array(list(islice(shuffled_data[V1], *mini_batch_ixs))), np.array(list(islice(shuffled_data[V2], *mini_batch_ixs)))],
                           [np.array(list(islice(shuffled_data[LABEL], *mini_batch_ixs)))])

    def evaluate(self, test_graph):
        """
        To evaluate the model
        :param test_graph:
        :type test_graph: networkx.Graph
        :return:
        :rtype:
        """
        edges_nx = test_graph.edges(data=True)
        num_edges = test_graph.number_of_edges()
        test_edges = [
            (self.node_2_ix[u], self.node_2_ix[v])
            for u, v, _ in edges_nx
        ]
        data = {
            V1: list(),
            V2: list(),
            LABEL: list()
        }
        for i in range(num_edges):
            v1, v2 = test_edges[i]
            data[V1].append(v1)
            data[V2].append(v2)
            data[LABEL].append(1.0)

            for _ in range(self.negative_ratio):
                data[V1].append(v1)
                while True:
                    v3 = self.node_alias_sampling.alias_sample()
                    if v3 != v2:
                        break
                data[V2].append(v3)
                data[LABEL].append(-1.0)
        print(self.model.evaluate(x=[np.array(data[V1]), np.array(data[V1])], y=[np.array(data[LABEL])]))

    def fetch_embedding_as_dict(self):
        """
        Fetches the embedding for each and every node in the graph

        :return:
        :rtype:
        """
        return {self.ix_2_node[ix]: embedding
                for ix, embedding in enumerate(self.embed.get_weights()[0])
                }

    def get_embedding(self, node_ix):
        """
        Gets the embedding corresponding to a particular node

        :param node_ix: index of node as captured in node_2_ix
        :type node_ix: int

        :return: embedding
        :rtype: numpy.ndarray
        """
        return self.embed.get_weights()[0][node_ix]
