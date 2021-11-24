import numpy as np
from common.functions import *
from common.util import *

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x

        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int((H + 2*self.pad - FH) / self.stride + 1)
        out_W = int((W + 2*self.pad - FW) / self.stride + 1)

        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = np.dot(self.col, self.col_W) + self.b

        out = out.reshape(N, out_h, out_W, -1).transpose(0,3,1,2)
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        x_shape = self.x.shape

        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        dcol = np.dot(dout, self.col_W.T)
        dW = np.dot(self.col.T, dout)
        self.dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
        self.db = np.sum(dout, axis=0)

        dx = col2im(dcol, x_shape, FH, FW, self.stride, self.pad)
        return dx

    def output_shape(self, input_shape):
        N, C, H, W = input_shape
        FN, C, FH, FW = self.W.shape
        out_H = (H - FH + 2*self.pad) / self.stride + 1
        out_W = (W - FW + 2*self.pad) / self.stride + 1
        out_shape = (N, FN, out_H, out_W)
        return out_shape


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class Flatten:
    def __init__(self):
        self.shape = None
    
    def forward(self, x):
        self.shape = x.shape

        out = x.reshape((x.shape[0], -1))
        return out

    def backward(self, dout):
        dx = dout.reshape(self.shape)
        return dx