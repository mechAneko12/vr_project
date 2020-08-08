import numpy as np
import sys
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from opt import SGD, EVE


class Affine:
    def __init__(self, weight=None, bias=None, input_size=None, hidden_size=None, weight_init_std=0.01, optimizer='sgd', learning_rate=1e-4):
        if weight.any() != None and bias.any() != None:
            self.W = np.array(weight)
            self.b = np.array(bias)
        else:
            self.W = weight_init_std * np.random.randn(input_size, hidden_size)
            self.b = np.zeros(hidden_size)
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

        opt_list = ['sgd', 'eve']
        if not optimizer in opt_list:
            print('optimizer name is invalid.')
            sys.exit(1)
        optimizers = [SGD(learning_rate), EVE(params_num = 2, alpha=learning_rate)]
        self.optimizer = optimizers[opt_list.index(optimizer)]

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

    def optimize(self, loss_=None):
        self.optimizer.main(self.W, self.dW, loss_)
        self.optimizer.main(self.b, self.db, loss_)
        return self

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class BatchNormalization:
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, optimizer='sgd', learning_rate=1e-4):
        self.input_shape = None
        self.momentum = momentum
        self.running_mu = None
        self.running_var = None

        self.gamma = gamma
        self.beta = beta

        self.dgamma = None
        self.dbeta = None

        self.x_mu = None
        self.var = None
        self.denom = None
        self.x_hat = None

        opt_list = ['sgd', 'eve']
        if not optimizer in opt_list:
            print('optimizer name is invalid.')
            sys.exit(1)
        optimizers = [SGD(learning_rate), EVE(params_num = 2, alpha=learning_rate)]
        self.optimizer = optimizers[opt_list.index(optimizer)]

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 3:
            N, C, L = x.shape
            x = x.reshape(N, -1)
        elif x.ndim == 4:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        
        if self.running_mu is None:
            N, D = x.shape
            self.running_mu = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = np.mean(x, axis=0)
            self.x_mu = x - mu
            self.var = np.mean(self.x_mu**2, axis=0)
            self.denom = np.sqrt(self.var + 1e-8)
            x_hat = self.x_mu / self.denom
            self.x_hat = x_hat

            self.running_mu = self.momentum * self.running_mu + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            x_hat = (x - self.running_mu) / np.sqrt(self.running_var + 1e-8)

        out = self.gamma * x_hat + self.beta
        return out.reshape(*self.input_shape)

    def backward(self, dout):
        if dout.ndim == 3:
            N, C, L = dout.shape
            dout = dout.reshape(N, -1)
        elif dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(self.x_hat * dout, axis=0)

        d_x_hat = dout * self.gamma
        pre_dx = self.denom * (d_x_hat - self.x_mu * np.mean(self.x_mu * d_x_hat, axis=0) / (self.var + 1e-8))
        dx = pre_dx - np.mean(pre_dx, axis=0)
        """
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.x_hat * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.denom
        dstd = -np.sum((dxn * self.x_mu) / (self.denom * self.denom), axis=0)
        dvar = 0.5 * dstd / self.denom
        dxc += (2.0 / self.batch_size) * self.x_mu * dvar
        dmu = np.sum(dxc, axis=0)
        dx_ = dxc - dmu / self.batch_size"""

        return dx.reshape(*self.input_shape)

    def optimize(self, loss_=None):
        self.optimizer.main(self.gamma, self.dgamma, loss_)
        self.optimizer.main(self.beta, self.dbeta, loss_)
        return self

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * self.dropout_ratio

    def backward(self, dout):
        return dout * self.mask


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]

        self.mask = (y == 0)
        y[self.mask] = 1e-15

        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss_ = self.cross_entropy_error(self.y, self.t)
        
        return self.loss_

    def backward(self, dout=1.0):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
