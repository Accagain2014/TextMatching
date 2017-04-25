#coding=utf-8

import numpy as np
import math
from scipy import sparse as sps

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


def oversample(X_ot,y,p=0.165):
    pos_ot = X_ot[y==1]
    neg_ot = X_ot[y==0]

    scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = sps.vstack([neg_ot, neg_ot]).tocsr()
        scale -=1
    neg_ot = sps.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = sps.vstack([pos_ot, neg_ot]).tocsr()
    y=np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]]=1.0
    print 'After oversample, the "is_duplicate" field mean: ', y.mean()
    return ot,y