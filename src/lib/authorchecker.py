import tensorflow as tf
from code2features import TokenType


class Checker(object):
    def __init__(self, index_size=TokenType.size):
        self.delta_input = tf.placeholder(shape=(None, index_size), dtype=tf.float32)
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

        x = tf.abs(tf.multiply(self.delta_input, self.weights))
        l = tf.reduce_sum(x, axis=1)
        self.answer = tf.subtract(1.0, tf.nn.sigmoid(tf.add(l, self.bias)))

    def save(self, sess, path):
        tf.train.Saver().save(sess, path)

    def restore(self, sess, path):
        tf.train.Saver().restore(sess, path)

    def get_input(self):
        return self.delta_input

    def get_answer(self):
        return self.answer
