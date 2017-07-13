#coding=utf-8

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cos_dis(x, y):
    '''
    Calculate cosine distance about vector x and y

    :param x: np.array liked, a vector, one dimension
    :param y: np.array liked, a vector, one dimension
    :return: cosine distance between vector x and vector y
    '''

    dot_mul = x * y
    ans = np.sum(dot_mul) / np.power(np.sum(x**2), 0.5) / np.power(np.sum(y**2), 0.5)

    '''
    ans_sk = cosine_similarity(x, y)

    assert ans - ans_sk < 1e-10
    '''

    return ans

