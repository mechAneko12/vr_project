import numpy as np

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
