import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

from line.model import LineBaseClass
from line.utils import V1, V2, LABEL, WEIGHT


def line_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_true * y_pred)))


class Line(LineBaseClass):
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
            input_dim=self.num_nodes,
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

    def run(self, epochs):
        """

        :param epochs:
        :type epochs:
        :return:
        :rtype:
        """
        logging.info("Compiling a model with Adam optimizer")
        self.model.compile(
            optimizer='adam',
            loss=line_loss,
            metrics=[tf.keras.metrics.Accuracy()]
        )

        logging.info("Fitting the model with batch size:{}, epochs:{}".format(self.batch_size, epochs))
        batch_gen = self.batch_size_gen(self.batch_size)
        samples_per_epoch = self.num_edges * (1+self.negative_ratio)
        steps_per_epoch = ((samples_per_epoch - 1) // self.batch_size + 1)
        self.model.fit(batch_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def evaluate(self, test_graph):
        v = {
            V1: list(),
            V2: list(),
            LABEL: list()
        }
        edges = [
            (self.node_2_ix[u], self.node_2_ix[v], _[WEIGHT])
            for u, v, _ in test_graph.edges(data=True)
        ]
        for v1, v2, w in edges:
            v[V1].append(int(v1))
            v[V2].append(v2)
            v[LABEL].append(float(w))
        test_dataset = self.dataset_gen(v[V1], v[V2], v[LABEL])
        print(self.model.evaluate(test_dataset))
