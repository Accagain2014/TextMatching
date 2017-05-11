#coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

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
