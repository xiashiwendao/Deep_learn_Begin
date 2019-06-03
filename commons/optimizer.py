import numpy as np


class Momentum:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):
        # 字典为空
        if not self.v:
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class SGD:
    # 初始化，lr：learn rate，学习率
    def __init__(self, lr=0.01):
        self.lr = 0.01

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = {}
    
    def update(self, params, grads):
        if not self.h:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key]/np.sqrt(self.h[key] + 1e-7)

