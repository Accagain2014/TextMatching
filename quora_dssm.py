#coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from scipy import sparse as sps
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from DSSM.dssm import DSSM
from helper.wordhash import WordHash
import tools
import config

def get_words(sentences):
    words = []
    for one_sen in sentences:
        one_sen = one_sen.strip().split()
        one_sen = map(lambda x: x.strip(), one_sen)
        words += one_sen
    return words


def quora_dssm(train_input_file, test_input_file):

    '''
    测试函数
    :return:
    '''
    seed = 2222

    train_ori = pd.read_csv(train_input_file)
    test_ori = pd.read_csv(test_input_file, nrows=1001)
    test_ori['is_duplicate'] = 0
    train_ori = train_ori[:1000]
    test = test_ori[:1000]

    print train_ori['is_duplicate'].value_counts()

    q = ['question1', 'question2']
    words = []

    wordhash = WordHash(words, load_from_file=True, \
                        dump_to_file=True, file='n_gram_term_index_mapping.pkl')

    train_ori_q1 =  wordhash.get_n_gram_count(train_ori[q[0]].values, is_dump=True, dump_file='result/train_q1_ngram_counting_matrix.pkl')
    train_ori_q2 = wordhash.get_n_gram_count(train_ori[q[1]].values, is_dump=True, dump_file='result/train_q2_ngram_counting_matrix.pkl')

    test_q1 = wordhash.get_n_gram_count(test[q[0]].values, is_dump=True, dump_file='result/test_q1_ngram_counting_matrix.pkl')
    test_q2 = wordhash.get_n_gram_count(test[q[1]].values, is_dump=True, dump_file='result/test_q2_ngram_counting_matrix.pkl')

    X = sps.hstack(
        [train_ori_q1, train_ori_q2]
    ).tocsr()
    y = train_ori['is_duplicate'].values[:]

    X_t = sps.hstack(
        [test_q1, test_q2]
    ).tocsr()
    y_t = test['is_duplicate'].values[:]

    '''
        Get origin train and test svm format file.
    '''
    dump_svmlight_file(X, y, 'result/train_ori_n_gram_counting_sparse_matrix.svm')
    dump_svmlight_file(X_t, y_t, 'result/test_n_gram_counting_sparse_matrix.svm')


    '''
        Get KFold train and valid dataset.
    '''
    skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
    for ind_tr, ind_te in skf:
        X_train = X[ind_tr]
        X_valid = X[ind_te]

        y_train = y[ind_tr]
        y_valid = y[ind_te]
        break

    X_train, y_train = tools.oversample(X_train.tocsr(), y_train, p=0.165)
    X_valid, y_valid = tools.oversample(X_valid.tocsr(), y_valid, p=0.165)

    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    dump_svmlight_file(X_train, y_train, 'result/oversample_train_n_gram_counting_sparse_matrix.svm')
    dump_svmlight_file(X_valid, y_valid, 'result/oversample_valid_n_gram_counting_sparse_matrix.svm')

    print 'Dump to svm format done.'


    '''
    for _ in q:
        train_ori[_] = train_ori[_].astype(str)
        test[_] = test_ori[_].astype(str)
        words += get_words(train_ori[_].values)
        words += get_words(test[_].values)

    print 'Sum words: ', len(words), ' sum diff words: ', len(set(words))

    wordhash = WordHash(words, load_from_file=True, load_file='n_gram_term_index_mapping.pkl', \
                        dump_to_file=True, dump_file='n_gram_term_index_mapping.pkl')


    split_point = int(0.7 * len(train_ori))
    train = train_ori[:split_point]
    valid = train_ori[split_point:]

    train_q1 = wordhash.get_n_gram_count(train[q[0]].values, is_dump=True, dump_file='result/train_q1_ngram_counting_matrix.pkl')
    train_q2 = wordhash.get_n_gram_count(train[q[1]].values, is_dump=True, dump_file='result/train_q2_ngram_counting_matrix.pkl')
    train_label = train['is_duplicate'].values

    valid_q1 = wordhash.get_n_gram_count(valid[q[0]].values, is_dump=True, dump_file='result/valid_q1_ngram_counting_matrix.pkl')
    valid_q2 = wordhash.get_n_gram_count(valid[q[1]].values, is_dump=True, dump_file='result/valid_q2_ngram_counting_matrix.pkl')
    valid_label = valid['is_duplicate'].values

    test_q1 = wordhash.get_n_gram_count(test[q[0]].values, is_dump=True, dump_file='result/test_q1_ngram_counting_matrix.pkl')
    test_q2 = wordhash.get_n_gram_count(test[q[1]].values, is_dump=True, dump_file='result/test_q2_ngram_counting_matrix.pkl')
    test_label = test['is_duplicate'].values


    print 'train shape: ', train_q1.shape, 'valid shape: ', valid_q1.shape
    print 'test shape: ', test_q1.shape
    n_gram_size = train_q1.shape[1]

    with tf.Graph().as_default():
        tf.set_random_seed(1)

        model = DSSM(hash_tokens_nums=n_gram_size, dnn_layer_nums=5, dnn_hidden_node_nums=10, feature_nums=10,
                batch_size=10, neg_nums=0, learning_rate=0.1, max_epochs=400, loss_kind='log_loss', w_init=0.01,
                     save_model_path='result/save-model', mlp_hidden_node_nums=32, mlp_layer_nums=2)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        np.random.seed(1)

        # query = np.random.rand(500, 30000)
        # doc = np.random.rand(500, 30000)
        # label = np.array([1, 0, 0, 0, 0] * 100)
        # model.set_positive_weights([1]*500)

        #print query
        #print doc
        #print label

        losses = model.fit(sess, train_q1, train_q2, train_label, valid_q1, valid_q2, valid_label, load_model=False)
        print 'Start to test. '

        test['is_duplicate'] = model.predict(sess, test_q1, test_q2, test_label)
        test[['test_id', 'is_duplicate']].to_csv('result/out.csv', index=False)
    '''

if __name__ == '__main__':

    train_file = config.train_file
    test_file = config.test_file

    quora_dssm(train_file, test_file)