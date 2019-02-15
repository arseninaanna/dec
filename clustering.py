from keras.engine.topology import Layer, InputSpec
import keras.backend as K
import math
import numpy as np
from scipy.spatial import distance
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans



class Clustering(Layer):
    def __init__(self, output_dim, input_dim=None, weights=None, **kwargs):
        self.centroids = self.output_dim = output_dim
        self.input_dim = input_dim
        self.initial_weights = weights
        #self.input_spec = [InputSpec(ndim=2)]

        if input_dim is not None:
            kwargs['input_shape'] = (input_dim,)
        super(Clustering, self).__init__(**kwargs)

    # todo
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

        super(Clustering, self).build(input_shape)

    def call(self, x, **kwargs):
        #sess = K.get_session()
        #q = soft_assignment(self.weights, K.eval(x))
        #return q
        q = 1.0 / (1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2)) ** 2 / 1.0)
        q = q ** ((1.0 + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim,
                  'weights': self.initial_weights}
        base_config = super(Clustering, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
    p = np.zeros((len(q), len(q.T)))
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
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w
