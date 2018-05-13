import tensorflow as tf
from code2features import TokenType


class Checker(object):
    def __init__(self, index_size=TokenType.size * TokenType.size):
        checker_classes = 2
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1, seed=1234567, dtype=tf.float32)

        self.weights = tf.get_variable(
            "weights",
            shape=(index_size, checker_classes),
            dtype=tf.float32,
            initializer=initializer
        )
        self.postweight = tf.get_variable(
            "postweight",
            shape=(checker_classes, 1),
            dtype=tf.float32,
            initializer=initializer
        )
        self.bias = tf.get_variable(
            "bias",
            shape=(checker_classes,),
            dtype=tf.float32,
            initializer=initializer
        )
        self.postbias = tf.get_variable(
            "postbias",
            shape=(checker_classes,),
            dtype=tf.float32,
            initializer=initializer
        )
        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver()

    def apply(self, value):
        x0 = tf.matmul(tf.abs(value), tf.abs(self.weights))
        x = tf.nn.sigmoid(tf.add(x0, self.bias))
        t0 = tf.matmul(x, self.postweight)
        t = tf.nn.sigmoid(tf.add(t0, self.postbias))
        return tf.subtract(1.0, tf.reduce_mean(x, axis=1))

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        self.saver.restore(sess, path)

    def get_input(self):
        return self.delta_input
