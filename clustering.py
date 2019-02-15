from keras.engine.topology import Layer, InputSpec
import keras.backend as K
import math
import numpy as np
from scipy.spatial import distance
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans


class Clustering(Layer):
    def __init__(self, n_clusters, input_dim=None, centroids=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Clustering, self).__init__(**kwargs)

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.initial_weights = centroids

    # todo
    def build(self, input_shape):
        assert len(input_shape) == 2
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, self.input_dim))

        self.clusters = self.add_weight((self.n_clusters, self.input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.weights is not None:
            self.set_weights(self.weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        q = soft_assignment(self.weights, x)
        return q

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_clusters


def get_centroids(x, count=10):
    kmeans = KMeans(n_clusters=count, n_init=20)
    kmeans.fit(x)

    return kmeans.cluster_centers_


def calculate_fj(q, j):
    return np.sum(q[j])


def soft_assignment(centroids, x):
    q = np.zeros(len(x), len(centroids))
    for i in range(len(x)):
        for j in range(len(centroids)):
            num = math.pow(1 + distance.euclidean(x[i], centroids[j]), -0.5)
            den = 0
            for k in range(len(centroids)):
                den += math.pow(1 + distance.euclidean(x[i], centroids[k]), -0.5)
            q[i][j] = num / den
    return q


def p_stat(q):
    p = np.zeros(len(q), len(q.T))
    for i in range(len(p)):
        for j in range(len(p.T)):
            fj = calculate_fj(q, i)
            num = math.pow(q[i][j], 2) / fj
            den = 0
            for k in range(len(q.T)):
                den += math.pow(q[i][k], 2) / fj  # here is a bug
            p[i][j] = num / den
    return p


def calculate_kl(p, q):
    kl = 0
    for i in range(len(q)):
        for j in range(len(q.T)):
            kl += p[i][j] * K.log(p[i][j] / q[i][j])
    return kl


def labels_delta(l_old, l_new):
    return (l_new == l_old).sum().astype(np.float32) / l_new.shape[0]


# todo: rewrite
def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w
