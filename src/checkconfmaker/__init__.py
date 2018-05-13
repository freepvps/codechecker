import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import lib
import argparse
import numpy as np
import tensorflow as tf
import random

from tensorflow.python.client import device_lib


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


def calc_metrics(real_answers, answers):
    tp = 0.00001
    tn = 0.00001
    fp = 0.00001
    fn = 0.00001

    for j, val in enumerate(answers):
        if real_answers[j] > 0.5:
            is_valid = 1 if val > 0.5 else 0
            tp += is_valid
            fn += 1 - is_valid
        else:
            is_valid = 1 if val < 0.5 else 0
            tn += is_valid
            fp += 1 - is_valid
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def random_group_valid(vecs, labels, prc, seed):
    random.Random(x=seed).shuffle(vecs)
    random.Random(x=seed).shuffle(labels)
    validation_size = int(len(vecs) * prc / 100)

    data_vecs_valid = vecs[:validation_size]
    data_labels_valid = labels[:validation_size]
    data_vecs = vecs[validation_size:]
    data_labels = labels[validation_size:]
    return data_vecs, data_labels, data_vecs_valid, data_labels_valid


def random_repository_valid(vecs, labels, prc, seed):
    reps = list(set(labels))
    random.Random(x=seed).shuffle(reps)
    validation_size = int(len(reps) * prc / 100)
    valid_reps = set(reps[:validation_size])

    data_vecs = [v for i, v in enumerate(vecs) if labels[i] not in valid_reps]
    data_labels = [v for i, v in enumerate(labels) if labels[i] not in valid_reps]
    data_vecs_valid = [v for i, v in enumerate(vecs) if labels[i] in valid_reps]
    data_labels_valid = [v for i, v in enumerate(labels) if labels[i] in valid_reps]

    return data_vecs, data_labels, data_vecs_valid, data_labels_valid


def main():
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    target_device = "/gpu:0" if get_available_gpus() else "/cpu:0"
    print(target_device)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, dest="dataset_dir", help="Input dataset path")
    parser.add_argument("-s", required=False, type=int, dest="files_count", help="Repository files count", default=10)
    parser.add_argument("-o", required=True, dest="output_file", help="Output model")
    parser.add_argument("-v", required=False, type=float, dest="validation_prc", help="Validation %", default=0.0)
    parser.add_argument("-r", required=False, type=int, dest="random_seed", help="Random seed for validation", default=1234567)
    parser.add_argument("-t", required=False, type=str, choices=["group", "repository"], dest="validation_type", help="Validation type", default="group")
    args = parser.parse_args()

    raw_data_vecs, raw_data_labels = load_data(args.dataset_dir, args.files_count)
    validation_selector = random_group_valid if args.validation_type == "group" else random_repository_valid

    data_vecs, data_labels, data_vecs_valid, data_labels_valid = validation_selector(
        raw_data_vecs,
        raw_data_labels,
        args.random_seed,
        args.validation_prc
    )

    deltas = [np.array(v1) - np.array(v2) for v1 in data_vecs for v2 in data_vecs]
    answers_raw = [1.0 if a1 == a2 else 0.0 for a1 in data_labels for a2 in data_labels]

    deltas_valid = [np.array(v1) - np.array(v2) for v1 in data_vecs_valid for v2 in data_vecs_valid]
    answers_raw_valid = [1.0 if a1 == a2 else 0.0 for a1 in data_labels_valid for a2 in data_labels_valid]

    with tf.Session() as sess:
        with tf.device(target_device):
            model = lib.authorchecker.Checker(len(deltas[0]))

            model_answers = model.apply(tf.constant(np.array(deltas), dtype=tf.float32))
            answers = tf.constant(answers_raw)
            loss = tf.reduce_mean(tf.pow(tf.subtract(model_answers, answers), 2.0))

            model_answers_valid = None
            if validation_size:
                model_answers_valid = model.apply(tf.constant(np.array(deltas_valid), dtype=tf.float32))

            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
            tf.global_variables_initializer().run()
            perfect_state = (0.0, 0.0, 0.0)
            for i in range(10000):
                _, loss_val, answer_val = sess.run((optimizer, loss, model_answers))
                accuracy, precision, recall = cur_state = calc_metrics(answers_raw, answer_val)
                print("{}. loss={}, accuracy={}, precision={}, recall={}".format(i, loss_val, accuracy, precision, recall))

                if validation_size:
                    answer_val_valid = sess.run(model_answers_valid)
                    accuracy_valid, precision_valid, recall_valid = cur_state = calc_metrics(answers_raw_valid, answer_val_valid)
                    print(
                        "{}. VALIDATION accuracy={}, precision={}, recall={}, perfect={}".format(
                            i,
                            accuracy_valid,
                            precision_valid,
                            recall_valid,
                            perfect_state
                        )
                    )
                if i % 100 == 0 and cur_state > perfect_state:
                    perfect_state = cur_state
                    model.save(sess, args.output_file)
        model.save(sess, args.output_file)


if __name__ == "__main__":
    main()
