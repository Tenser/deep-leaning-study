import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict

class MultiLayerNet:

    def __init__(self, input_size, hidden_size_list, output_size, weight_init=0):
        self.step_num = len(hidden_size_list) + 1

        self.params = {}

        if weight_init == 0:
            self.params["W1"] = 0.02 * np.random.randn(input_size, hidden_size_list[0])
            for i in range(1, len(hidden_size_list)):
                self.params["W"+str(i+1)] = 0.02 * np.random.randn(hidden_size_list[i-1], hidden_size_list[i])
            self.params["W"+str(len(hidden_size_list)+1)] = 0.02 * np.random.randn(hidden_size_list[len(hidden_size_list)-1], output_size)

        elif weight_init == 1:
            self.params["W1"] = np.random.randn(input_size, hidden_size_list[0]) / np.sqrt(input_size)
            for i in range(1, len(hidden_size_list)):
                self.params["W"+str(i+1)] = np.random.randn(hidden_size_list[i-1], hidden_size_list[i]) / np.sqrt(hidden_size_list[i-1])
            self.params["W"+str(len(hidden_size_list)+1)] \
                = np.random.randn(hidden_size_list[len(hidden_size_list)-1], output_size) / np.sqrt(input_size)

        elif weight_init == 2:
            self.params["W1"] = np.random.randn(input_size, hidden_size_list[0]) / np.sqrt(input_size) * np.sqrt(2)
            for i in range(1, len(hidden_size_list)):
                self.params["W"+str(i+1)] \
                    = np.random.randn(hidden_size_list[i-1], hidden_size_list[i]) / np.sqrt(hidden_size_list[i-1]) * np.sqrt(2)
            self.params["W"+str(len(hidden_size_list)+1)] \
                = np.random.randn(hidden_size_list[len(hidden_size_list)-1], output_size) / np.sqrt(input_size) * np.sqrt(2)

        for i in range(len(hidden_size_list)):
                self.params["b"+str(i+1)] = np.zeros(hidden_size_list[i])
        self.params["b"+str(len(hidden_size_list)+1)] = np.zeros(output_size)

        self.layers = OrderedDict()
        for i in range(len(hidden_size_list)):
            self.layers["Affine"+str(i+1)] = Affine(self.params["W"+str(i+1)], self.params["b"+str(i+1)])
            self.layers["ReLU"+str(i+1)] = ReLU()
        self.layers["Affine"+str(len(hidden_size_list)+1)] \
            = Affine(self.params["W"+str(len(hidden_size_list)+1)], self.params["b"+str(len(hidden_size_list)+1)])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        x = self.predict(x)
        loss = self.lastLayer.forward(x, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y==t) / float(y.shape[0])

    """
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads
    """

    def gradient(self, x, t):
        loss = self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for i in range(self.step_num):
            grads["W"+str(i+1)] = self.layers["Affine"+str(i+1)].dW
            grads["b"+str(i+1)] = self.layers["Affine"+str(i+1)].db
        return grads, loss

    """
    def gradient_descent(self, x, t, lr, step_num):
        train_size = x.shape[0]
        batch_size = 100
        
        for i in range(step_num):
            
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x[batch_mask]
            t_batch = t[batch_mask]

            grads = self.gradient(x_batch, t_batch)
            for key in ("W1", "W2", "b1", "b2"):
                self.params[key] -= lr * grads[key]
            
            #print(self.loss(x_batch, t_batch))
    """

    def predict_res(self, x):
        y = self.predict(x)
        return np.argmax(y)
