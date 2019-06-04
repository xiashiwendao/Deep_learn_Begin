import os, sys
sys.path.append(os.getcwd())
from commons.layers import *
import numpy as np
from collections import OrderedDict


class MultiLayerNetwork:
    def __init__(self, input_size, hidden_size_list, output_size, activation="relu", weight_decay_lambda=0):
        # 初始化参数
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_size = len(hidden_size_list)
        self.output_size = output_size
        self.activation = activation
        self.weight_decay_lambda = weight_decay_lambda

        # 初始化weight权重参数
        self.params = {}
        self.__init_weight(activation)
        # 生成层
        # 注意这里一定是要OrderedDict，否则因为乱序问题导致无法正常按照顺序处理数据
        self.layers = OrderedDict()
        activation_fn_dic = {"sigmod": Sigmoid, "relu": Relu}
        # 构建隐藏层
        for idx in range(1, self.hidden_layer_size+ 1):
            W, b = self.params["W" + str(idx)], self.params["b" + str(idx)]
            self.layers["Affine" + str(idx)] = Affine(W, b)
            self.layers["activation_fn" + str(idx)] = activation_fn_dic[activation.lower()]()
        # 构建输出层
        idx += 1
        W, b = self.params["W" + str(idx)], self.params["b" + str(idx)]
        self.layers["Affine" + str(idx)] = Affine(W, b)
        self.lastlayer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_layer_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        total_layer_size = len(all_layer_size_list)
        scale = weight_init_std
        # 遍历各个层，初始化weight以及b值
        for idx in range(1, total_layer_size):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_layer_size_list[idx - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_layer_size_list[idx - 1])
            # 计算并存储各个层的weight以及b值
            self.params["W" + str(idx)] = scale * np.random.randn(all_layer_size_list[idx-1], all_layer_size_list[idx])
            self.params["b" + str(idx)] = np.zeros(all_layer_size_list[idx])
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        weigh_decay = 0
        # 注意这里+2,是因为还要把lastLayer也给加上
        for idx in range(1, self.hidden_layer_size+2):
            W = self.params["W" + str(idx)]
            weigh_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        self.lastlayer.forward(y, t) + weigh_decay

    def gradient(self, x, t):
        # 构建正向链
        self.loss(x, t)
        # 输出层（softmax）求导
        dout = 1
        dout = self.lastlayer.backward(dout)
        # 隐藏层求导
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.hidden_layer_size + 2):
            grads["W" + str(idx)] = self.layers["Affine" + str(idx)].dW + self.weight_decay_lambda * self.layers["Affine" + str(idx)].W
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

        return grads
