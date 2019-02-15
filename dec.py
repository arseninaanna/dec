import clustering as cl
import utils
from clustering import Clustering
from keras.models import Sequential
from keras.optimizers import SGD


def get_model(encoder, x):
    clusters = 10
    learning_rate = 0.01
    momentum = 0.9

    output = encoder.predict(x)
    centroids = cl.get_centroids(output, clusters)

    clustering_layer = Clustering(n_clusters=clusters, weights=centroids, name='clustering')
    model = Sequential([
        encoder,
        clustering_layer
    ])
    model.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
    # model.compile(loss=cl.calculate_kl, optimizer=SGD(lr=learning_rate, momentum=momentum))

    return model


def train_dec(dec_model, x, y=None):
    conv_thr = 0.1
    cur_conv = 1

    while cur_conv > conv_thr:
        # todo
        Z = dec_model.predict(x)


if __name__ == '__main__':
    X, Y = utils.get_mnist()
    print("MNIST loaded")

    encoder = utils.load_model('encoder')
    print("Encoder loaded")

    try:
        dec_model = utils.load_model('dec')
        print('DEC loaded')
    except EnvironmentError:
        dec_model = get_model(encoder, X)

        print('Training DEC')
        train_dec(dec_model, X)

        print('Saving DEC')
        utils.save_model('dec', dec_model)

    print()
