import os, sys
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, x_train) = load_mnist(normalize=True, one_hot_label=True)
print(np.shape(x_train))