import types
import math
import numpy as np
import code2features
from gensim.models.word2vec import Word2Vec


def softmax(v):
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=0)


def distance_to_probability(v, sigm=1.32):
    v = np.asarray(v)
    res = softmax(-v * v / (sigm * sigm))
    min_prob = 10**(-7)
    for i in range(len(res)):
        if res[i] < min_prob:
            res[i] = min_prob
    return res


def make_index(sentences):
    """
    :type sentences: list[basestring]
    :param sentences: input code
    :return: list of numpy vectors
    :rtype: list[np.ndarray]
    """
    features = [map(str, code2features.extract_features(sentence)) for sentence in sentences]
    w2v_model = Word2Vec(features, size=30, window=20, min_count=2, workers=1)

    vectors = map(np.array, w2v_model.wv.vectors)
    labels = [int(w2v_model.wv.index2word[i]) for i in range(len(vectors))]

    features_size = code2features.TokenType.size
    features_vecs = [None] * features_size
    for i, v in enumerate(vectors):
        features_vecs[labels[i]] = v

    dist_matrix = np.zeros((features_size, features_size), dtype=np.float32)
    dists_list = []
    for i, v1 in enumerate(features_vecs):
        for j, v2 in enumerate(features_vecs):
            if features_vecs[i] is None and features_vecs[j] is None:
                dist_matrix[i][j] = 0.0
            elif features_vecs[i] is None or features_vecs[j] is None:
                dist_matrix[i][j] = 100000.0
            else:
                dist_matrix[i][j] = np.linalg.norm(v1 - v2)
                dists_list.append(dist_matrix[i][j])
    sigm = math.sqrt(np.var(dists_list))

    probs = [distance_to_probability(v, sigm) for v in dist_matrix]
    flatten_probs = np.array(probs).flatten()
    res = np.ndarray.tolist(flatten_probs)

    return res
