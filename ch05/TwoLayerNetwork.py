from collections import OrderedDict
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