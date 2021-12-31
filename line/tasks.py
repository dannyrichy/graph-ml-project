import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from line.model import LineBaseClass


def _line(dataset, size, embedding_dim, learning_rate=0.001, num_epochs=1000):
    inputs = {
        "target": layers.Input(name="target", shape=(), dtype="int32"),
        "context": layers.Input(name="context", shape=(), dtype="int32")
    }
    label = layers.Input(name='label', shape=(), dtype=tf.int32)
    embed_item = layers.Embedding(
        input_dim=size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    target_embeddings = embed_item(inputs["target"])
    context_embeddings = embed_item(inputs["context"])
    logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embeddings, context_embeddings]
    )
    model = keras.Model(inputs=inputs, outputs=logits)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=tf.nn.sigmoid_cross_entropy_with_logits,
    )
    model.fit(dataset, epochs=num_epochs)
    return model


def _prepare_dataset(targets, contexts, labels, weights, batch_size=1024):
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def run(graph):
    """

    :param graph:
    :type graph: nx.Graph
    :return:
    :rtype:
    """
    logging.info("Running the model")
    batch_size = 1024
    num_negative_sample = 10
    N_ITER = 10

    logging.info("I don't even know what I coded here")
    base = LineBaseClass(graph=graph)
    v = base.generate_batch_size(bs=batch_size, num_negative_sample=num_negative_sample)

    logging.info("Preparing the dataset I guess")
    dataset = _prepare_dataset(v['v1'], v['v2'], v['label'], weights=v['weight'], batch_size=batch_size)
    logging.info("Yay! Model is being trained")
    model = _line(dataset=dataset, size=batch_size * (num_negative_sample + 1), embedding_dim=128)
