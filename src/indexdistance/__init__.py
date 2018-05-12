import json
import sys
import numpy as np
import os


def load_vec(path):
    with open(path) as f:
        return json.load(f)


labels = map(os.path.basename, sys.argv[1:])
vecs = map(np.array, map(load_vec, sys.argv[1:]))


for i, v1 in enumerate(vecs):
    for j, v2 in enumerate(vecs):
        dist = np.linalg.norm(v1 - v2)
        print("{} - {} distance: {}".format(labels[i], labels[j], dist))