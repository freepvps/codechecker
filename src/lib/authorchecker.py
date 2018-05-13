import tensorflow as tf
from code2features import TokenType


class Checker(object):
    def __init__(self, index_size=TokenType.size * TokenType.size):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1, seed=1234567, dtype=tf.float32)

        self.weights = tf.get_variable(
            "weights",
            shape=(index_size,),
            dtype=tf.float32,
            initializer=initializer
        )
        self.bias = tf.get_variable(
            "bias",
            shape=(),
            dtype=tf.float32,
            initializer=initializer
        )
        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver()

    def apply(self, value):
        x0 = tf.reduce_sum(tf.abs(tf.multiply(value, self.weights)), axis=1)
        x = tf.nn.sigmoid(tf.add(x0, self.bias))
        return tf.subtract(1.0, x)

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        self.saver.restore(sess, path)

    def get_input(self):
        return self.delta_input
