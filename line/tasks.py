import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from line.model import LineBaseClass

LEARNING_RATE = 0.001


def line_loss(y_true, y_pred):
    return -keras.backend.mean(keras.backend.log(keras.backend.sigmoid(y_true * y_pred)))


class Line1(LineBaseClass):
    def __init__(self, graph, batch_size=128, negative_ratio=5, embedding_dim=128):
        """

        :param graph:
        :type graph: networkx.Graph
        :param batch_size:
        :type batch_size: int
        :param negative_ratio: Number of negative samples to construct
        :type negative_ratio: int
        :param embedding_dim:
        :type embedding_dim: int
        """
        logging.info("Initialising the base class")
        super().__init__(graph, negative_ratio)
        self.batch_size = batch_size

        # Defining the input block
        logging.info("Defining the input block")
        self.input = {
            "target": layers.Input(name="target", shape=(), dtype="int32"),
            "context": layers.Input(name="context", shape=(), dtype="int32")
        }

        # Defining the embedding block
        logging.info("Embedding layer initialised")
        self.embed = layers.Embedding(
            input_dim=self.graph.number_of_nodes(),
            output_dim=embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="embeddings",
        )

        # Applying the embedding to the inputs
        logging.info("Application of embedding layer")
        self.embedded_input = {
            key: self.embed(value)
            for key, value in self.input.items()
        }

        # Defining the output block
        logging.info("Output layer defined")
        self.output = layers.Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keepdims=False), name='first_order')([self.embedded_input['target'], self.embedded_input['context']])

        # Defining the model
        self.model = Model(inputs=[self.input['target'], self.input['context']], outputs=[self.output])

    def dataset_gen(self, targets, contexts, labels):
        """

        :param targets:
        :type targets:
        :param contexts:
        :type contexts:
        :param labels:
        :type labels:
        :return:
        :rtype:
        """
        inputs = {
            "target": targets,
            "context": contexts,
        }
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.shuffle(buffer_size=self.batch_size * 2)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def run(self, no_iter):
        """

        :param no_iter:
        :type no_iter:
        :return:
        :rtype:
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=line_loss
        )
        logging.info("Running the iteration")
        for i in range(no_iter):
            logging.info("Iteration {}".format(i))
            v = self.generate_batch_size(bs=self.batch_size)
            dataset = self.dataset_gen(v['v1'], v['v2'], v['label'])
            self.model.fit(dataset)

    def evaluate(self, test_graph):
        v = {
            'v1': list(),
            'v2': list(),
            'label': list()
        }
        edges = [
            (self.node_2_ix[u], self.node_2_ix[v], _['weight'])
            for u, v, _ in test_graph.edges(data=True)
        ]
        for v1, v2, w in edges:
            v['v1'].append(int(v1))
            v['v2'].append(v2)
            v['label'].append(float(w))
        test_dataset = self.dataset_gen(v['v1'], v['v2'], v['label'])
        print(self.model.evaluate(test_dataset))
