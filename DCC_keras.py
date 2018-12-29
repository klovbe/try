from .config import *
from utils import *
from com_param import *
from dataset import *
from DCCLoss import *
from metrics import *

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Dense, Lambda, Subtract, merge, Dropout, BatchNormalization, Activation
from keras.models import Model, model_from_json, Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as K

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



name = 'zeisel'
train_set, train_labels, graph = prepare_data(name)
nsamples = len(train_labels)
dims = [train_set.shape[1], 500, 500, 2000, 10]
batch_size = 256
dr_rate = 0.2
n_iters_pretrain = 2000
epochs_pretrain = n_iters_pretrain//batch_size
nu1 = 0.0
nu2 = 0.0
optimizer = SGD(lr=0.1, momentum=0.99)

weights, nconn = cal_weight(train_set, graph)
dcc = DCC()
dcc.pretrain()
dcc.ae_build()
dcc.ae_v_build()
Z = dcc.encoders.predict(train_set)
_sigma2, _lambda, _delta, _delta2, lmdb, lmdb_data = computeHyperParams(graph, Z, weights)
oldassignment = np.zeros(len(graph))
stopping_threshold = int(math.ceil(cfg.STOPPING_CRITERION * float(len(graph))))
n_iters = 2000
epochs = n_iters//batch_size
# double_x = np.append(train_set, train_set, axis=1)
optimizer_g = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.)
dcc.pre_model.compile(optimizer=optimizer_g, loss=[weighted_mse_x, weighted_mse_x, rho_mse_y(_sigma2)],
                       loss_weights=[1, 1, _lambda])
# This is the actual Algorithm
flag = 0
for epoch in range(epochs):
    dcc.pre_model.fit_generator(
        generator=batch_generator_sdne(train_set, graph, weights, nconn, batch_size=batch_size, shuffle=True, beta=1.0),
        epochs=1, steps_per_epoch=min((graph.shape[0] // batch_size), nsamples))
    Z = dcc.encoders.predict(train_set)

    change_in_assign = 0
    assignment = -np.ones(nsamples)
    # logs clustering measures only if sigma2 has reached the minimum (delta2)
    if flag:
        index, ari, ami, nmi, acc, n_components, assignment = computeObj(Z, graph, _delta, train_labels, nsamples)
        print('acc = %.5f, nmi = %.5f, ari = %.5f, ari = %.5f' % (acc, nmi, ari, ami))
        change_in_assign = np.abs(oldassignment - index).sum()
        oldassignment[...] = index

        # As long as the change in label assignment < threshold, DCC continues to run.
        # Note: This condition is always met in the very first epoch after the flag is set.
        # This false criterion is overwritten by checking for the condition twice.
        if change_in_assign > stopping_threshold:
            flag += 1
        if flag == 4:
            break

    if ((epoch + 1) % 4 == 0):
        _sigma2 = max(_delta2, _sigma2 / 2)
        if _sigma2 == _delta2 and flag == 0:
            # Start checking for stopping criterion
            flag = 1

class DCC():
    def __int__(self):
        pass


    def pretrain(self):
        X_train_tmp = train_set
        self.trained_encoders = []
        self.trained_decoders = []
        for i in range(len(dims)-1):
            print('Pre-training the layer: Input {} -> {} -> Output {}'.format(dims[i], dims[i+1], dims[i]))
            # Create AE and training
            print(i)
            ae = Sequential()
            if i == 0:
                ae.add(Dropout(dr_rate))
                ae.add(Dense(dims[i+1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2), name='encoder_%d' % i))
                ae.add(Dense(dims[i], input_dim=dims[i+1], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2), name='decoder_%d' % i))
            elif i == len(dims)-2:
                ae.add(Dropout(dr_rate))
                ae.add(Dense(dims[i+1], input_dim=dims[i], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2), name='encoder_%d' % i))
                ae.add(Dense(dims[i], input_dim=dims[i+1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2), name='decoder_%d' % i))
            else:
                ae.add(Dropout(dr_rate))
                ae.add(Dense(dims[i+1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2), name='encoder_%d' % i))
                ae.add(Dense(dims[i], input_dim=dims[i+1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2), name='decoder_%d' % i))
            ae.compile(loss='mean_squared_error', optimizer='adam')
            ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=epochs_pretrain)
            ae.summary()
            # Store trainined weight
            self.trained_encoders.append(ae.layers[1])
            self.trained_decoders.append(ae.layers[2])
            # Update training data
            encoder = Model(ae.input, ae.layers[1].output)
            X_train_tmp = encoder.predict(X_train_tmp)

    def ae_build(self):
        print('Fine-tuning')
        self.decoders = Sequential()
        self.encoders = Sequential()
        # autoencoders.add(InputLayer(input_shape=(dims[0],)))
        # encoders.add(InputLayer(input_shape=(dims[0],)))
        for encoder in self.trained_encoders[:-1]:
        #     autoencoders.add(encoder)
            self.encoders.add(encoder)
        for decoder in self.trained_decoders[:-1][::-1]:
            self.decoders.add(decoder)
        self.x = Input(shape=(dims[0],), name='input')
        self.h = self.encoders(self.x)
        self.y = self.decoders(self.h)
        self.autoencoders = Model(inputs=self.x, outputs=self.y)

    def ae_v_build(self):
        print('Fine-tuning')
        decoder_v = Sequential()
        encoder_v = Sequential()
        # autoencoders.add(InputLayer(input_shape=(dims[0],)))
        # encoders.add(InputLayer(input_shape=(dims[0],)))
        for encoder in self.trained_encoders:
            #     autoencoders.add(encoder)
            encoder_v.add(encoder)
        for decoder in self.trained_decoders[::-1]:
            decoder_v.add(decoder)
        x_v = Input(shape=(dims[0],), name='input')
        v_v = encoder_v(x_v)
        y_v = decoder_v(v_v)
        self.autoencoder_v = Model(inputs=x_v, outputs=y_v)
        self.autoencoder_v.save_weights('./cluster/{}_autoencoder_pretrain.h5'.format(name))

    def train_ae(self):
        self.autoencoders.compile(optimizer='adam', loss='mse')
        self.autoencoders.fit(train_set, train_set, epochs=1000, batch_size=256)
        self.autoencoder_v.save_weights('./cluster/{}_autoencoder_pure.h5'.format(name))

    def load_pretrain_weights(self):
        self.autoencoder_v.load_weights('./cluster/{}_autoencoder_pretrain.h5'.format(name))

    def build_graph(self):
        x_in = Input(shape=(2 * dims[0],), name='x_in')
        x1 = Lambda(lambda x: x[:, 0:dims[0]], output_shape=(dims[0],))(x_in)
        x2 = Lambda(lambda x: x[:, dims[0]:2 * dims[0]], output_shape=(dims[0],))(x_in)
        # Process inputs
        x_hat1 = self.autoencoders(x1)
        x_hat2 = self.autoencoders(x2)
        y1 = self.encoders(x1)
        y2 = self.encoders(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2])
        y_diff = Subtract()([y2, y1])
        self.pre_model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])




