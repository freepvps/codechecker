import tensorflow as tf
import numpy as np


def train(vecs, answers):
    vec_dim = len(vecs[0])
    dvecs = [np.array(v1) - np.array(v2) for v1 in vecs for v2 in vecs]
    ans_vec = [0.0 if a1 == a2 else 1.0 for a1 in answers for a2 in answers]

    input_vecs = tf.placeholder(shape=(None, vec_dim), dtype=tf.float32)
    output_ans = tf.placeholder(shape=(None,), dtype=tf.float32)

    initializer = tf.truncated_normal_initializer(mean=1.0, stddev=0.001, seed=1234567, dtype=tf.float32)

    weights = tf.get_variable(
        "weights",
        shape=(vec_dim,),
        dtype=tf.float32,
        initializer=initializer
    )
    bias = tf.get_variable(
        "bias",
        shape=(),
        dtype=tf.float32,
        initializer=initializer
    )

    x = tf.abs(tf.multiply(input_vecs, weights))
    l = tf.reduce_sum(x, axis=1)
    result = tf.nn.sigmoid(tf.add(l, bias))
    loss = tf.reduce_mean(tf.pow(tf.subtract(result, output_ans), 2.0))

    opt = tf.train.AdamOptimizer(learning_rate=1)
    grads = opt.compute_gradients(loss)
    optimizer = opt.minimize(loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(1000):
            _, loss_val, grad_val, w_val = sess.run([optimizer, loss, grads, weights], feed_dict={
                input_vecs: dvecs,
                output_ans: ans_vec
            })
            print(i, loss_val, w_val)
        return sess.run((weights, bias))


