# coding: utf-8
import os
import sys
class SGD:
    # 初始化，lr：learn rate，学习率
    def __init__(self, lr=0.01):
        self.lr = 0.01

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


