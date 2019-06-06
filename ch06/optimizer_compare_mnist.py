import os, sys
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
import logging

from commons.optimizer import *
from commons.MultiLayerNetwork import MultiLayerNetwork
from matplotlib import pyplot as plt
from commons.utils import smooth_curve

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# 1. 获取到数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 2. 设置参数
train_size = np.shape(x_train)[0] # 获取训练样本数
batch_size = 128 # 每次批量训练的样本数
max_iteration = 1000 # 迭代（优化）次数

# 3. 定义优化器列表（用于后续遍历）
optimizers = {} 
optimizers["SGD"] = SGD()
optimizers["Momentum"] = Momentum()
optimizers["AdaGrad"] = AdaGrad()
optimizers["Adam"] = Adam()

# 4. 构建神经网络集合，每个神经网络使用一种优化器
networks = {}
train_loss = {} # 损失值字典，会记录各个优化器的优化过程中损失函数的值
for key in optimizers.keys():
    networks[key] = MultiLayerNetwork(input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []

# 5. 进行迭代
for i in range(max_iteration):
    # 随机选出batch_size大小的样本进行训练
    batch_indeies = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_indeies]
    t_batch = t_train[batch_indeies]

    # 遍历各个神经网络（优化器），通过批量样本进行训练，并记录损失函数值
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)

        train_loss[key].append(loss)

    if i % 100 == 0:
        print("iteration: ", i)
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key, ": ", loss)

print("train_loss:\n", train_loss)

# 6. 画图
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iteration)
logging.info("x列表内容: %s", x)
for key in optimizers.keys():
    smoothed = smooth_curve(train_loss[key])
    logging.info("smoothed内容: %s", markers[key])
    plt.plot(x, smoothed, marker=markers[key], markevery=100, label=key)
plt.xlabel("iteration")
plt.ylabel("loss")
#plt.ylim(0, 1)
plt.legend()
plt.show()