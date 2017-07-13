#coding=utf-8

import numpy as np
import math
from scipy import sparse as sps
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

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


def data_iterator(orig_X, orig_y=None, orig_label=None, batch_size=10, shuffle=False, is_normalize=False):
    '''

    :param orig_X:
    :param orig_y:
    :param orig_label:
    :param batch_size:
    :param shuffle:
    :return:
    '''

    # Optionally shuffle the data before training
    if shuffle:
        indices = np.random.permutation(orig_X.shape[0])
        data_X = orig_X[indices]
        data_y = orig_y[indices]
        data_label = orig_label[indices]
    else:
        data_X = orig_X
        data_y = orig_y
        data_label = orig_label
    ###
    total_processed_examples = 0
    total_steps = int(np.ceil(data_X.shape[0]) / float(batch_size))
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start : batch_start + batch_size]
        y = data_y[batch_start : batch_start + batch_size]
        label = orig_label[batch_start : batch_start + batch_size]
        '''
        if is_sparse:
            yield x.toarray(), y.toarray(), label
        else:
            yield x, y, label
        '''
        if is_normalize:
            yield normalize(x, axis=0), normalize(y, axis=0), label
        else:
            yield x, y, label
        total_processed_examples += x.shape[0]
    # Sanity check to make sure we iterated over all the dataset as intended
    #assert total_processed_examples == data_X.shape[0], 'Expected {} and processed {}'.format(data_X.shape[0], total_processed_examples)
