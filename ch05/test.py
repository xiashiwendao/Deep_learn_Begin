# coding: utf-8
import sys, os
sys.path.append(os.getcwd())

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
print("before first layer parames: ", network.layers['Affine1'].W[0])
network.params["W1"] = None
print("after first layer parames: ", network.layers['Affine1'].W[0])


# iters_num = 10000
# train_size = x_train.shape[0]
# batch_size = 100
# learning_rate = 0.1

# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# # 梯度
# #grad = network.numerical_gradient(x_batch, t_batch)
# grad = network.gradient(x_batch, t_batch)

# print("before first layer parames: ", network.layers['Affine2'].W[0])
# # 更新
# for key in ('W1', 'b1', 'W2', 'b2'):
#     network.params[key] -= learning_rate * grad[key]
# print("grad[key]: ", grad[key])
# print("after  first layer parames: ", network.layers['Affine2'].W[0])
from collections import OrderedDict

class A:
    def __init__(self, W, b):
        self.params={}
        self.params["W1"] = np.zeros(3)
        self.params["b1"] = 6
        self.layers = OrderedDict()
        self.layers["building1"] = B(self.params["W1"], self.params["b1"])
        self.W = W
        self.b = b
        
class B:
    def __init__(self, W, b):
        self.W = W
        self.b = b

a = A(1,2)
print(a.layers["building1"].b)
a.params["b1"] = 5
print(a.layers["building1"].b)
print(a.params["W1"])