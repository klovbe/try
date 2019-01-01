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

def extend_graph(graph_df, new_column_list):
    # dim = len(new_column_list)
    # array = np.zeros((dim, dim), dtype=np.int32)
    # columns = list(graph_df.columns)
    # edge_list = []
    # column2id = dict(zip(new_column_list, range(dim)))
    # for ix, row in graph_df.iterrows():
    #     n_id = column2id[ix]
    #     for col in columns:
    #         value = row[col]
    #         col_id = column2id[col]
    #         if value > 0.0:
    #             array[n_id, col_id] = 1
    # new_graph_df = pd.DataFrame(array, index=new_column_list, columns=new_column_list)
    print("extending graph")
    ppi_gene_names = list(graph_df.columns)
    print("original ppi length: {}".format(len(ppi_gene_names)))
    cell_gene_names = new_column_list
    print("original gene number: {}".format(len(cell_gene_names)))
    gene_names = []
    for col in ppi_gene_names:
        if col not in cell_gene_names:
            continue
        gene_names.append(col)
    print("overlap gene number of ppi: {}".format(len(gene_names)))
    graph_df = graph_df.ix[gene_names, gene_names]
    values = graph_df.values
    dim = len(new_column_list)
    column2id = dict(zip(new_column_list, range(dim)))
    rows, cols = np.where(values > 0.0)
    edge_indexs = list( zip(rows, cols) )
    new_edge_indexs = [ [column2id[gene_names[x]], column2id[gene_names[y]] ] \
                        for (x, y) in edge_indexs
                      ]
    new_index_array = np.array(new_edge_indexs, dtype=np.int32)
    row_index, col_index = new_index_array[:, 0], new_index_array[:, 1]
    dim = len(new_column_list)
    new_graph = np.zeros((dim, dim), dtype=np.int32)
    new_graph[ row_index, col_index ] = 1.0
    new_graph_df = pd.DataFrame(new_graph, index=new_column_list, columns=new_column_list, dtype=np.int32)
    # new_graph = new_graph + np.diag(np.ones(shape=new_graph.shape[0]))
    return new_graph_df

def save_imputation(train_set, decoder_out, name):
    mask_data = train_set == 0.0
    mask_data = np.float32(mask_data)
    # decoder_out = autoencoders.predict(train_set)
    decoder_out_replace = mask_data * decoder_out + train_set
    df_raw = pd.DataFrame(decoder_out)
    df_raw.to_csv('./cluster/{}_pure_autoencoder.csv'.format(name), index=None, float_format='%.4f')
    df_replace = pd.DataFrame(decoder_out_replace)
    df_replace.to_csv('./cluster/{}_pure_autoencoder.csv'.format(name), index=None, float_format='%.4f')


if __name__ == "__main__":
    # fire.Fire()
    graph_df = pd.DataFrame([ [1,0, 1], [0, 1, 1], [1, 0, 1] ], index=["a", "b", "c"], columns=["a", "b", "c"])
    new_graph_df = extend_graph(graph_df, ["d", "a", "b", "c", "h"])
    print(graph_df)
    print(new_graph_df)