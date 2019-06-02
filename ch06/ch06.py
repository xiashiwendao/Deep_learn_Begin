import os
import sys
import numpy as np
sys.path.append("D:\\practicespace\\github\\Deep_learn_Begin")
from commons.TwoLayerNetwork import TwoLayerNetwork
from commons.SGD import SGD
from dataset.mnist import load_mnist

network = TwoLayerNetwork(784, 100, 10)
optimizer = SGD()
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
step = 5
loop_time = int(1000/5)
print("shape of x_train: ", np.shape(x_train))
print("shape of t_train: ", np.shape(t_train))
print(x_train)
print("caculating...")
def optimizeGrad(optimizer):
    for index in range(loop_time):
        x_batch = x_train[index * step : (index+1) * step]
        t_batch = t_train[index * step : (index+1) * step]
        grads = network.gradient(x_batch, t_batch)
        params = network.params
        optimizer.update(params, grads)

optimizeGrad(optimizer)
print("completed")