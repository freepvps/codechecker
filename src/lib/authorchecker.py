import tensorflow as tf
from code2features import TokenType


class Checker(object):
    def __init__(self, index_size=TokenType.size*TokenType.size):
        checker_classes = 3
        self.delta_input = tf.placeholder(shape=(None, index_size), dtype=tf.float32)
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1, seed=1234567, dtype=tf.float32)

        self.weights = tf.get_variable(
            "weights",
            shape=(index_size,checker_classes),
            dtype=tf.float32,
            initializer=initializer
        )
        self.postweight = tf.get_variable(
            "postweight",
            shape=(checker_classes,1),
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

        # x = tf.reduce_sum(tf.abs(tf.multiply(self.delta_input, self.weights)), axis=1)
        # l = tf.nn.sigmoid(tf.add(x, self.bias))
        # self.answer = tf.subtract(1.0, l)
        x0 = tf.matmul(tf.abs(self.delta_input), tf.abs(self.weights))
        x = tf.nn.sigmoid(tf.add(x0, self.bias))
        t0 = tf.matmul(x, self.postweight)
        t = tf.nn.sigmoid(tf.add(t0, self.postbias))

        self.answer = tf.subtract(1.0, tf.reduce_mean(t, axis=1))
        self.saver = tf.train.Saver()

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        self.saver.restore(sess, path)

    def get_input(self):
        return self.delta_input

    def get_answer(self):
        return self.answer
