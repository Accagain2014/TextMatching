import sys
sys.path.append('../helper')

import tensorflow as tf
import pandas as pd
import numpy as np
import tools
import wordhash


class CDSSM(object):
    pass

if __name__ == '__main__':
    a = np.array(range(12)).reshape(3, 4)
    print a
    print tools.softmax(a)