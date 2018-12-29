import torch
import torch.nn as nn
import numpy as np
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Dense, Lambda, Subtract, merge, Dropout, BatchNormalization, Activation
from keras.models import Model, model_from_json, Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as K

class DCCWeightedELoss(nn.Module):
    def __init__(self, size_average=True):
        super(DCCWeightedELoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, outputs, weights):
        out = (inputs - outputs).view(len(inputs), -1)
        out = torch.sum(weights * torch.norm(out, p=2, dim=1)**2)

        assert np.isfinite(out.data.cpu().numpy()).all(), 'Nan found in data'

        if self.size_average:
            out = out / inputs.nelement()

        return out

class DCCLoss(nn.Module):
    def __init__(self, nsamples, ndim, initU, size_average=True):
        super(DCCLoss, self).__init__()
        self.dim = ndim
        self.nsamples = nsamples
        self.size_average = size_average
        self.U = nn.Parameter(torch.Tensor(self.nsamples, self.dim))
        self.reset_parameters(initU+1e-6*np.random.randn(*initU.shape).astype(np.float32))

    def reset_parameters(self, initU):
        assert np.isfinite(initU).all(), 'Nan found in initialization'
        self.U.data = torch.from_numpy(initU)

    def forward(self, enc_out, sampweights, pairweights, pairs, index, _sigma1, _sigma2, _lambda):
        centroids = self.U[index]

        out1 = torch.norm((enc_out - centroids).view(len(enc_out), -1), p=2, dim=1) ** 2
        out11 = torch.sum(_sigma1 * sampweights * out1 / (_sigma1 + out1))

        out2 = torch.norm((centroids[pairs[:, 0]] - centroids[pairs[:, 1]]).view(len(pairs), -1), p=2, dim=1) ** 2

        out21 = _lambda * torch.sum(_sigma2 * pairweights * out2 / (_sigma2 + out2))

        out = out11 + out21

        if self.size_average:
            out = out / enc_out.nelement()

        return out

def weighted_mse_x(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    return K.sum(
        K.square(y_pred * y_true[:, :-1]),
        axis=-1) / y_true[:, -1]

def weighted_mse_y(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
    y_pred: Contains y2 - y1
    y_true: Contains s12
    '''
    min_batch_size = K.shape(y_true)[0]
    return K.reshape(
        K.sum(K.square(y_pred), axis=-1),
        [min_batch_size, 1]
    ) * y_true

def rho_mse_y(mu):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
    y_pred: Contains y2 - y1
    y_true: Contains s12
    '''
    def loss(y_true, y_pred):
        min_batch_size = K.shape(y_true)[0]
        tmp = K.reshape(
            K.sum(K.square(y_pred), axis=-1),
            [min_batch_size, 1]
        )
        return y_true * mu * tmp / (mu + tmp)

    return loss
