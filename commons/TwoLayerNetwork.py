from collections import OrderedDict
import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Sigmod:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
        
    def backward(self, d):
        dx = d * (1 - self.out) * self.out
        
        return dx

class Affine:
    def __init__(self, w, b):
        self.x = None
        self.W = w
        self.b = b
        self.dW = None
        self.db = None
    
    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        self.x = x
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx

# y代表学习结果，t代表真实标签
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None
    
    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        # 这里为什么是y和t呢？y代表学习结果，t代笔真实的分类
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
        
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx

class TwoLayerNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_layer_size, hidden_layer_size)
        self.params["b1"] = np.zeros(hidden_layer_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_layer_size, output_layer_size)
        self.params["b2"] = np.zeros(output_layer_size)
        
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        
        self.lastlayers = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayers.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        # 为什么是argmax呢？
        y = np.argmax(y, axis=1)
        # 同问
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_w, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_w, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_w, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_w, self.params["b2"])
        
        return grads
        
    def gradient(self, x, t):
        # build network
        self.loss(x, t)
        dout = 1
        dout = self.lastlayers.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        
        return grads