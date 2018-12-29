import numpy as np
import pandas as pd
from utils import *

def row_normal(data, factor=1e6):
    row_sum = np.sum(data, axis=1)
    row_sum = np.expand_dims(row_sum, 1)
    div = np.divide(data, row_sum)
    div = np.log(1 + factor * div)
    return div

def load_newdata(train_datapath, metric='pearson', gene_scale=True, data_type='count', trans=True ):
    print("make dataset from {}...".format(train_datapath))
    df = pd.read_csv(train_datapath, sep=",", index_col=0)
    if trans:
        df = df.transpose()
    print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
    if data_type == 'count':
        df = row_normal(df)
        # df = sizefactor(df)
    elif data_type == 'rpkm':
        df = np.log(df + 1)
    if gene_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data=data, columns=df.columns)
    return df.values

def batch_generator_sdne(X, graph, weights, nconn, batch_size, shuffle, beta=1):
    row_indices = graph[:,0]
    col_indices = graph[:,1]
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter: batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :]
        X_batch_v_j = X[col_indices[batch_index], :]
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)
        B_i[X_batch_v_i != 0] = beta
        B_j = np.ones(X_batch_v_j.shape)
        B_j[X_batch_v_j != 0] = beta
        X_ij = weights(batch_index)
        deg_i = np.sum(nconn[row_indices[batch_index]], 1).reshape((batch_size, 1))
        deg_j = np.sum(nconn[col_indices[batch_index]], 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        OutData = [a1, a2, X_ij.T]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def cal_weight(X, graph):
    numpairs = graph.shape[0]
    numsamples = X.shape[0]
    # Creating pairwise weights and individual sample sample for reconstruction loss term
    R = csr_matrix((np.ones(numpairs, dtype=np.float32), (graph[:, 0], graph[:, 1])), shape=(numsamples, numsamples))
    R = R + R.transpose()
    nconn = np.squeeze(np.array(np.sum(R, 1)))
    weights = np.average(nconn) / np.sqrt(nconn[graph[:, 0]] * nconn[graph[:, 1]])
    # pairs = np.hstack((graph, np.atleast_2d(weights).transpose()))
    return  weights, nconn


def prepare_data(name):
    train_datapath = '/home/xysmlx/data/filter_data/{}_count.csv'.format(name)
    x = load_newdata(train_datapath)
    labelpath = '/home/xysmlx/data/label_big/{}_label.csv'.format(name)
    # labelpath = '/home/xysmlx/data/label_big/zeisel_label.csv'
    from sklearn.preprocessing import LabelEncoder
    labeldf = pd.read_csv(labelpath, header=0, index_col=0)
    y = labeldf.values
    y = y.transpose()
    y_name = np.squeeze(y)
    if not isinstance(y, (int, float)):
        y = LabelEncoder().fit_transform(y_name)
    n_clusters = len(np.unique(y))
    print("has {} clusters:".format(n_clusters))
    print("orginal cluster proportion: {}".format(np.bincount(y)))

    from time import time
    t0 = time()
    graph = m_knn(x, 10)
    print("construct mknn graph takes time : ", time() - t0)
    np.savetxt('./cluster/{}_mknn_graph.csv'.format(name), graph, delimiter=',')
    return x, y, graph