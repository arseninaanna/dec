from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import math
import utils


def get_centroids(x):
    kmeans = KMeans(n_init=20)
    kmeans.fit(x)

    return kmeans.cluster_centers_


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


def calculate_fj(q, i):
    fj = 0
    for k in range(len(q.T)):
        fj += q[i][k]
    return fj


def p_stat(q):
    p = np.zeros(len(q), len(q.T))
    for i in range(len(p)):
        for j in range(len(p.T)):
            fj = calculate_fj(q, i)
            num = math.pow(q[i][j], 2) / fj
            den = 0
            for k in range(len(q.T)):
                den += math.pow(q[i][k], 2) / fj
            p[i][j] = num / den
    return p


def calculate_kl(p, q):
    kl = 0
    for i in range(len(q)):
        for j in range(len(q.T)):
            kl += p[i][j] * math.log10(p[i][j] / q[i][j])
    return kl


def train_dec(encoder, x):
    conv = 0.1
    cur_conv = 1

    while cur_conv > conv:
        Z = encoder.predict(x)
        centroids = get_centroids(Z)
        q = soft_assignment(centroids, Z)
        p = p_stat(q)
        KL = calculate_kl(p, q)

    return {}  # todo: complete


if __name__ == '__main__':
    X, Y = utils.get_mnist()
    encoder = utils.load_model('encoder')

    print("Data loaded")

    dec_model = train_dec(encoder, X)
