from keras.engine.topology import Layer, InputSpec
import keras.backend as K
import keras.losses
import math
import numpy as np
from scipy.spatial import distance
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans


class Clustering(Layer):
    def __init__(self, output_dim, input_dim=None, weights=None, prediction=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initial_weights = weights
        self.initial_prediction = prediction

        if input_dim is not None:
            kwargs['input_shape'] = (input_dim,)
        super(Clustering, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

        super(Clustering, self).build(input_shape)

    def call(self, x, **kwargs):
        # sess = K.get_session()
        # q = soft_assignment(self.weights, K.eval(x))
        # return q
        q = 1.0 / (1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2)) ** 2 / 1.0)
        q = q ** ((1.0 + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'weights': self.initial_weights
        }
        base_config = super(Clustering, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# helps to initialize DEC
def get_centroids(x, count=10):
    km = KMeans(n_clusters=count, n_init=20)
    pred = km.fit_predict(x)

    return km.cluster_centers_, pred


# DEC actual predictor
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


# DEC loss function
def calculate_kl(p, q):
    return K.sum(p * K.log(p / q), axis=-1)


# calculates target values. this is our 'y_true'
def p_stat(q):
    n, m = q.shape
    q_square = np.power(q, 2)  # calculates q**2 for each element
    fj_vector = np.sum(q, axis=0)  # calculates f() for each column

    p = np.zeros((n, m))
    for i in range(n):
        num = q_square[i] / fj_vector
        den = np.sum(q_square[i] / fj_vector)

        p[i] = num / den

    return p


# helps to detect training convergence
def labels_delta(l_old, l_new):
    data_size = l_new.shape[0]
    diff_count = np.sum(l_new == l_old)

    return 1 - (diff_count / data_size)


# for metrics only
def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w


# workaround for keras loading bugs
keras.losses.calculate_kl = calculate_kl
