import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import lib
import argparse
import numpy as np
import tensorflow as tf


def load_data(path, files_count):
    dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    reps = [lib.load_repository(os.path.join(path, x)) for x in dirs]

    labels = []
    blocks = []
    for i, rep in enumerate(reps):
        for j in range(0, len(rep), files_count):
            print("make index for {}-{}".format(i, j / files_count))
            sub_rep = rep[j:j + files_count]
            if len(sub_rep) == files_count:
                labels.append(dirs[i])
                blocks.append(lib.index_maker.make_index(sub_rep))
    return blocks, labels


if __name__ == "__main__":
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']


    target_device = "/gpu:0" if get_available_gpus() else "/cpu:0"
    print(target_device)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, dest="dataset_dir", help="Input dataset path")
    parser.add_argument("-s", required=False, dest="files_count", help="Repository files count", default=10)
    parser.add_argument("-o", required=True, dest="output_file", help="Output model")
    args = parser.parse_args()

    data_vecs, data_labels = load_data(args.dataset_dir, args.files_count)

    deltas = [np.array(v1) - np.array(v2) for v1 in data_vecs for v2 in data_vecs]
    answers_raw = [1.0 if a1 == a2 else 0.0 for a1 in data_labels for a2 in data_labels]
    mean_answers = np.mean(answers_raw)

    with tf.device(target_device):
        answers = tf.constant(answers_raw)

        model = lib.authorchecker.Checker(len(deltas[0]))

        loss = tf.reduce_mean(tf.pow(tf.subtract(model.get_answer(), answers), 2.0))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(1000):
                _, loss_val = sess.run((optimizer, loss), feed_dict={
                    model.get_input(): deltas
                })
                print(i, loss_val, mean_answers)
            model.save(sess, args.output_file)
