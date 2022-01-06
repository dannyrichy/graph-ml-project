import tensorflow as tf
from sklearn.model_selection import train_test_split


def logistic_regression(x, W, b):
    return tf.nn.softmax(tf.matmul(x, W) + b)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def cross_entropy(y_pred, y_true, num_classes):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


def classification(embeddings, labels, batch_size=1024, embedding_dim=128, num_classes=2):
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=4, random_state=4)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    W = tf.Variable(tf.ones([embedding_dim, num_classes]), name="weight")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")

    optimizer = tf.optimizers.SGD(0.01)

    for step, (batch_x, batch_y) in enumerate(train_data, 1):
        with tf.GradientTape() as g:
            pred = logistic_regression(batch_x, W, b)
            loss = cross_entropy(pred, batch_y, num_classes)

        gradients = g.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))

    pred = logistic_regression(x_test, W, b)
    print("Test Accuracy: %f" % accuracy(pred, y_test))
