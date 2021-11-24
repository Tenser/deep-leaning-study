import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict

class MultiConvNet:

    def __init__(self, input_dim, conv_params, pool_params, hidden_size, output_size, weight_init="std"):
        self.conv_num = len(conv_params)

        self.params = {}

        self.params["W1"] = \
            0.02 * np.random.randn(conv_params[0]["filter_num"], input_dim[0], \
                conv_params[0]["filter_size_h"], conv_params[0]["filter_size_w"])
        self.params["b1"] = np.zeros(conv_params[0]["filter_num"])

        conv_output_size_h = (input_dim[1] - conv_params[0]["filter_size_h"] + 2*conv_params[0]["pad"]) \
            / conv_params[0]["stride"] + 1
        conv_output_size_w = (input_dim[2] - conv_params[0]["filter_size_w"] + 2*conv_params[0]["pad"]) \
            / conv_params[0]["stride"] + 1
        pool_output_dim = [conv_params[0]["filter_num"], conv_output_size_h/pool_params[0]["pool_h"] \
            , conv_output_size_w/pool_params[0]["pool_w"]]

        for i in range(1, self.conv_num):
            self.params["W"+str(i+1)] = \
                0.02 * np.random.randn(conv_params[i]["filter_num"], conv_params[i-1]["filter_num"], \
                    conv_params[i]["filter_size_h"], conv_params[i]["filter_size_w"])
            self.params["b"+str(i+1)] = np.zeros(conv_params[i]["filter_num"])

            conv_output_size_h = (pool_output_dim[1] - conv_params[i]["filter_size_h"] + 2*conv_params[i]["pad"]) \
                / conv_params[i]["stride"] + 1
            conv_output_size_w = (pool_output_dim[2] - conv_params[i]["filter_size_w"] + 2*conv_params[i]["pad"]) \
                / conv_params[i]["stride"] + 1
            pool_output_dim = [conv_params[i]["filter_num"], conv_output_size_h/pool_params[i]["pool_h"] \
                , conv_output_size_w/pool_params[i]["pool_w"]]

        last_pool_output_size = int(pool_output_dim[0] * pool_output_dim[1] * pool_output_dim[2]) 

        c = { "std":0.02, "xiaos": 1 / np.sqrt(last_pool_output_size), "He": np.sqrt(2) / np.sqrt(last_pool_output_size)}
        self.params["W"+str(self.conv_num+1)] = c[weight_init] * np.random.randn(last_pool_output_size, hidden_size)
        self.params["b"+str(self.conv_num+1)] = np.zeros(hidden_size)

        c = { "std":0.02, "xiaos": 1 / np.sqrt(hidden_size), "He": np.sqrt(2) / np.sqrt(hidden_size)}
        self.params["W"+str(self.conv_num+2)] = c[weight_init] * np.random.randn(hidden_size, output_size)
        self.params["b"+str(self.conv_num+2)] = np.zeros(output_size)

        
        self.layers = OrderedDict()

        for i in range(self.conv_num):
            self.layers["Conv"+str(i+1)] = \
                Convolution(self.params["W"+str(i+1)], self.params["b"+str(i+1)], stride=conv_params[i]["stride"], pad=conv_params[i]["pad"])
            self.layers["ReLU"+str(i+1)] = ReLU()
            self.layers["Pool"+str(i+1)] = Pooling(pool_params[i]["pool_h"], pool_params[i]["pool_w"], pool_params[i]["stride"])
            
        self.layers["Flatten"+str(self.conv_num+1)] = Flatten()
        self.layers["Affine"+str(self.conv_num+1)] \
            = Affine(self.params["W"+str(self.conv_num+1)], self.params["b"+str(self.conv_num+1)])
        self.layers["ReLU"+str(self.conv_num+1)] = ReLU()

        self.layers["Affine"+str(self.conv_num+2)] \
            = Affine(self.params["W"+str(self.conv_num+2)], self.params["b"+str(self.conv_num+2)])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
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

    def gradient(self, x, t):
        loss = self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for i in range(self.conv_num):
            grads["W"+str(i+1)] = self.layers["Conv"+str(i+1)].dW
            grads["b"+str(i+1)] = self.layers["Conv"+str(i+1)].db
        grads["W"+str(self.conv_num+1)] = self.layers["Affine"+str(self.conv_num+1)].dW
        grads["b"+str(self.conv_num+1)] = self.layers["Affine"+str(self.conv_num+1)].db
        grads["W"+str(self.conv_num+2)] = self.layers["Affine"+str(self.conv_num+2)].dW
        grads["b"+str(self.conv_num+2)] = self.layers["Affine"+str(self.conv_num+2)].db
        return grads, loss

    def predict_res(self, x):
        y = self.predict(x)
        return np.argmax(y)
