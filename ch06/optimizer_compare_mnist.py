import os, sys
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
from commons.optimizer import *
from commons.MultiLayerNetwork import MultiLayerNetwork
from matplotlib import pyplot as plt
from commons.utils import smooth_curve
# 1. 获取到数据
(x_train, t_train), (x_test, x_train) = load_mnist(normalize=True, one_hot_label=True)

# 2. 设置参数
train_size = np.shape(x_train)[0] # 获取训练样本数
batch_size = 128 # 每次批量训练的样本数
max_iteration = 2000 # 迭代（优化）次数

# 3. 定义优化器列表（用于后续遍历）
optimizers = {} 
optimizers["SGD"] = SGD(lr = 0.7)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr = 0.3)

# 4. 构建神经网络集合，每个神经网络使用一种优化器
network = {}
train_loss = {} # 损失值字典，会记录各个优化器的优化过程中损失函数的值
for optimizer in optimizers.keys():
    network[optimizer] = MultiLayerNetwork(input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[optimizer] = []

# 5. 进行迭代
for i in range(max_iteration):
    # 随机选出batch_size大小的样本进行训练
    batch_indeies = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_indeies]
    t_batch = t_train[batch_indeies]
    # 遍历各个神经网络（优化器），通过批量样本进行训练，并记录损失函数值
    for key in optimizers.keys():
        optimizer = optimizers[key]
        grads = optimizer.gradient(x_batch, t_batch)
        optimizer.update(network[key].params, grads)
        loss = optimizer.loss(x_batch, t_batch)

        train_loss[key].append(loss)
# 6. 画图
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iteration)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), markers=markers[key], markevery=100, label=key)

plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

