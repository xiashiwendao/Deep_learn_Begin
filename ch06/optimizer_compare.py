from collections import OrderedDict
import sys, os
sys.path.append("C:\\Users\\wenyang.zhang\\Documents\\MySpace\\practice\\github\\Deep_learn_Begin")
from commons.optimizer import SGD
from commons.optimizer import Adam
from commons.optimizer import AdaGrad
from commons.optimizer import Momentum
import numpy as np
from matplotlib import pyplot as plt

def f(x, y):
    return x ** 2 / 10 + y * y

def df(x, y):
    return 2 * x / 10, 2 * y

init_pos = (-7.0, 2.0)
params = {}
params["x"], params["y"] = init_pos[0], init_pos[1]

grads = {}
grads["x"], grads["y"] = 0, 0

optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr = 0.7)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr = 0.3)
idx = 1
# 遍历优化器
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params["x"], params["y"] = init_pos[0], init_pos[1]
    # 构建优化器
    for i in range(30):
        x_history.append(params["x"])
        y_history.append(params["y"])

        grads["x"], grads["y"] = df(params["x"], params["y"])
        optimizer.update(params, grads)
    
    # 生成测试数据
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # 等高线
    mask = Z > 7
    Z[mask] = 0

    # 画图
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, "o-", color="red")
    plt.plot(0, 0, "+")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contour(X, Y, Z)

plt.show()