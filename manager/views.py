from django.shortcuts import render
from django.http.response import JsonResponse
from django.http import HttpResponse
from .models import Wavelet, Weight
import datetime
from django.core import serializers
import json
import numpy as np
import urllib.request

from collections import OrderedDict
import time
import copy

import sys
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Create your views here.
#page funnction
def index(request):
    template_name = "index.html" # templates以下のパスを書く
    return render(request,template_name)


#classes
class DWT_N:
    def __init__(self, n):
        if n < 4:
            print("n must be (n>=4).", file=sys.stderr)
            sys.exit(1)
        self.n = n
        self.fil = None

    def filter(self):
        sq3 = np.sqrt(3); fsq2 = 4.0*np.sqrt(2);  #N = 32      #N = 2^n
        c0 = (1 + sq3)/fsq2;    c1 = (3 + sq3)/fsq2             #Daubechies 4 coeff
        c2 = (3 - sq3)/fsq2;    c3 = (1 - sq3)/fsq2
        nend = 4
        nd = copy.copy(self.n)
        fil = []
        while nd >= nend:
            _fil = np.zeros((nd,nd))
            nd //= 2
            for ind in range(nd):
                _fil[ind*2][ind*2] = c0
                _fil[ind*2][ind*2+1] = c1
                _fil[ind*2+1][ind*2] = c3
                _fil[ind*2+1][ind*2+1] = -c2
                if ind == nd -1:
                    _fil[ind*2][0] = c2
                    _fil[ind*2][1] = c3
                    _fil[ind*2+1][0] = c1
                    _fil[ind*2+1][1] = -c0
                else:
                    _fil[ind*2][ind*2+2] = c2
                    _fil[ind*2][ind*2+3] = c3
                    _fil[ind*2+1][ind*2+2] = c1
                    _fil[ind*2+1][ind*2+3] = -c0
            fil.append(_fil)
        self.fil = fil
        return fil
    
    @staticmethod
    def daube4(f, nd, fil):
        f_tmp = f[:nd]
        f_tmp = np.dot(fil, f_tmp)
        nd_2 = nd//2
        for ind, tmp in enumerate(f_tmp):
            if ind % 2 == 0:
                f[ind//2] = tmp
            else:
                f[ind//2+nd_2] = tmp

    def main(self, f):
        nend = 4
        nd = copy.copy(self.n)
        i = 0
        while nd >= nend:
            self.daube4(f, nd, self.fil[i])
            nd //= 2
            i += 1


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


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def main(self, x, dx, loss_=None):
        delta = self.learning_rate * dx
        x -= delta
        return x

class EVE:
    def __init__(self, params_num, alpha=1e-3, beta_1=0.9, beta_2=0.999, beta_3=0.999, c = 10.0, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.c = c
        self.epsilon = epsilon

        self.m = []
        self.v = []
        self.pre_loss = []
        self.d_tilda = []
        for l in range(params_num):
            self.m.append(0.0)
            self.v.append(0.0)
            self.pre_loss.append(None)
            self.d_tilda.append(None)
        self.params_num = params_num
        self.param_index = 0
    
    def main(self, x, dx, loss_):
        self.m[self.param_index] = self.beta_1 * self.m[self.param_index] + (1 - self.beta_1) * dx
        m_hat = self.m[self.param_index] / (1 - self.beta_1)
        self.v[self.param_index] = self.beta_2 * self.v[self.param_index] + (1 - self.beta_2)* (dx * dx)
        v_hat = self.v[self.param_index] / (1 - self.beta_2)

        if self.d_tilda[self.param_index] != None:
            d = np.abs(loss_ - self.pre_loss[self.param_index]) / min(loss_, self.pre_loss[self.param_index])
            d_hat = np.clip(d, 1 / self.c, self.c)
            self.d_tilda[self.param_index] = self.beta_3 * self.d_tilda[self.param_index] + (1 - self.beta_3) * d_hat
        else:
            self.d_tilda[self.param_index] = 1
        
        x -= self.alpha * m_hat / (self.d_tilda[self.param_index] * (np.sqrt(v_hat) + self.epsilon))

        self.pre_loss[self.param_index] = loss_

        self.param_index += 1
        if self.param_index == self.params_num:
            self.param_index = 0

        return x

#predicted_classを返すfunction
json_open = open('staticfiles/jsonfiles/weight_data.json', 'r')
weights = json.load(json_open)
input_size = int(weights['input_size'])
hidden_size = int(weights['hidden_size'])
output_size = int(weights['output_size'])

layers = OrderedDict()
for lay_name in weights['layer_name']:
    if lay_name[0] == "D":
        layers[lay_name] = Dropout(dropout_ratio=0.5)
    elif lay_name[0] == "A":
        layers[lay_name] = Affine(weight=np.array(weights[lay_name]), bias=np.array(weights[lay_name+'_bias']))
    elif lay_name[0] == "B":
        layers[lay_name] = BatchNormalization()
    elif lay_name[0] == "R":
        layers[lay_name] = Relu()

N = weights['N']
def return_class(request):
    global layers, N
    emg_arr_raw = request.POST.getlist('emg_arr[]')
    d = {"predicted_class": np.array([float(x) for x in emg_arr_raw]).reshape(-1,8)}
    return JsonResponse(d)
    emg_arr = np.empty((0, 8))
    for row_raw in emg_arr_raw:
        row_raw.split(",")
        row = np.array([[int(x) for x in row_raw.split(",")]])
        emg_arr = np.concatenate([emg_arr, row], axis=0)
    
    dwt = DWT_N(N)
    dwt.filter()
    wavelet = np.empty((0, (N-2)*8))
    for i in range(emg_arr.shape[0] - N):
        f = copy.copy(emg_arr[i:i+N][:])
        dwt.main(f)
        wavelet = np.concatenate([wavelet, f[:][2:].T.reshape(1,(N-2)*8)], axis=0)
    predicted_data = predict(layers, wavelet, train_flg=False)
    predicted_class = predicted_data.argmax(axis=1)
    
    d = {"predicted_class": str(predicted_class)}
    return JsonResponse(d)

def predict(layers, x, train_flg=True):
    for layer in layers.values():
        if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
            x = layer.forward(x, train_flg)
        else:
            x = layer.forward(x)
    return x


