#coding=utf-8

import numpy as np
import pandas as pd


def data_iterator(orig_X, orig_y=None, orig_label=None, batch_size=10, shuffle=False):
    # Optionally shuffle the data before training
    if shuffle:
        indices = np.random.permutation(len(orig_X))
        data_X = orig_X[indices]
        data_y = orig_y[indices] if np.any(orig_y) else None
        data_label = orig_label[indices]
    else:
        data_X = orig_X
        data_y = orig_y
        data_label = orig_label
    ###
    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start : batch_start + batch_size]
        y = data_y[batch_start : batch_start + batch_size]
        label = orig_label[batch_start : batch_start + batch_size]
        yield x, y, label
        total_processed_examples += len(x)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X),
                                                                                          total_processed_examples)