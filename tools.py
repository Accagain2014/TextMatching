#coding=utf-8

import numpy as np
import math

def softmax(x):
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    return x


def sigmoid(x):
    x = 1. / (1 + np.exp(-x))
    return x


def sigmoid_grad(f):
    f = f * (1 - f)
    return f

def xavier_init(n1, n2):
    return math.sqrt(6.0/(n1+n2))