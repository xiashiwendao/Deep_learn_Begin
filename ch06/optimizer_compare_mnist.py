import os, sys
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
from commons.optimizer import SGD
from commons.optimizer import Momentum
from commons.optimizer import AdaGrad
from commons.optimizer import Adam
import MulLayerNetwork

def f(x, y):
    return x ** 2 / 20 + y ** 2

def df(x, y):
    return 2 * x /20, 2 * y

(x_train, t_train), (x_test, x_train) = load_mnist(normalize=True, one_hot_label=True)
print(np.shape(x_train))

train_size = np.shape(x_train)[0]
batch_size = 128
# 构建优化器
optimizers = {}
optimizers["SGD"] = SGD()
optimizers["Momentum"] = Momentum()
optimizers["AdaGrad"] = AdaGrad()
optimizers["Adam"] = Adam()
# 下面基于梯度下降，进行迭代优化
for i in range(2000):
    batch_indeies = np.random.choice(train_size, batch_size)
    batch_x = x_train[batch_indeies]
    batch_t = t_train[batch_indeies]
    network = {}
    # 遍历所有的优化器，构建多层网络
    for key in optimizers.keys():
        network[key] = MulLayerNetwork()
        optimzer.update(params, grads)