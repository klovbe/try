import pandas as pd
import numpy as np
import pandas as pd

import math

import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csr_matrix, triu, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial import distance

from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from time import time

from scipy.stats import *
from scipy.spatial.distance import *





def m_knn(X, k, measure='euclidean'):
    """
    This code is taken from:
    https://bitbucket.org/sohilas/robust-continuous-clustering/src/
    The original terms of the license apply.
    Construct mutual_kNN for large scale dataset

    If j is one of i's closest neighbors and i is also one of j's closest members,
    the edge will appear once with (i,j) where i < j.

    Parameters
    ----------
    X (array) 2d array of data of shape (n_samples, n_dim)
    k (int) number of neighbors for each sample in X
    measure (string) distance metric, one of 'cosine' or 'euclidean'
    """

    samples = X.shape[0]
    batch_size = 10000
    b = np.arange(k+1)
    b = tuple(b[1:].ravel())

    z = np.zeros((samples, k))
    weigh = np.zeros_like(z)

    # This loop speeds up the computation by operating in batches
    # This can be parallelized to further utilize CPU/GPU resource

    for x in np.arange(0, samples, batch_size):
        start = x
        end = min(x+batch_size, samples)

        w = distance.cdist(X[start:end], X, measure)

        y = np.argpartition(w, b, axis=1)

        z[start:end, :] = y[:, 1:k + 1]
        weigh[start:end, :] = np.reshape(w[tuple(np.repeat(np.arange(end-start), k)),
                                           tuple(y[:, 1:k+1].ravel())], (end-start, k))
        del w

    ind = np.repeat(np.arange(samples), k)

    P = csr_matrix((np.ones((samples*k)), (ind.ravel(), z.ravel())), shape=(samples, samples))
    Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))

    Tcsr = minimum_spanning_tree(Q)
    P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
    P = triu(P, k=1)

    V = np.asarray(find(P)).T
    return V[:, :2].astype(np.int32)

def plot_ae(X_embedded, name):
    if X_embedded.shape[1] > 2:
        X_embedded = TSNE(n_components=2).fit_transform(X_embedded)
    # fig = plt.gcf()
    # plt.savefig("k-means.png")
    plt.figure(figsize=(12, 10))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=train_labels)
    plt.colorbar()
    plt.show()

def save_imputation(train_set, decoder_out, name):
    mask_data = train_set == 0.0
    mask_data = np.float32(mask_data)
    # decoder_out = autoencoders.predict(train_set)
    decoder_out_replace = mask_data * decoder_out + train_set
    df_raw = pd.DataFrame(decoder_out)
    df_raw.to_csv('./cluster/{}_pure_autoencoder.csv'.format(name), index=None, float_format='%.4f')
    df_replace = pd.DataFrame(decoder_out_replace)
    df_replace.to_csv('./cluster/{}_pure_autoencoder.csv'.format(name), index=None, float_format='%.4f')