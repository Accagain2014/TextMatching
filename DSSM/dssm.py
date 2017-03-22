#coding=utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import time

import utils

class DSSM(object):
    '''
        Impletement DSSM Model in the Paper:  Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    '''
    def __init__(self, hash_tokens_nums=3000, dnn_layer_nums=1, dnn_hidden_node_nums=50, feature_nums=50,
                batch_size=10, neg_nums=4, learning_rate=0.5, max_epochs=200):
        '''
            paras:
                hash_tokens_nums: word hash后词的个数
                dnn_layer_nums: dnn的层数
                dnn_hidden_node_nums: dnn的结点个数
                feature_nums: 最终输出的特征的个数
                batch_size: 每个batch的大小
                neg_nums: 负样本的个数
                learning_rate: 学习率
                max_epoch: 迭代次数
        '''
        self.hash_token_nums = hash_tokens_nums
        self.dnn_layer_nums = dnn_layer_nums
        self.dnn_hidden_node_nums = dnn_hidden_node_nums
        self.feature_nums = feature_nums
        self.batch_size = batch_size
        self.neg_nums = neg_nums
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        '''
            query and doc 使用不同的网络结构，像论文中提到的那样
        '''
        self.input_q = tf.placeholder(tf.float32, shape=[batch_size, self.hash_token_nums]) # sample_nums, word_nums, hash_tokens_nums
        self.input_doc = tf.placeholder(tf.float32, shape=[batch_size, self.hash_token_nums]) # sample_nums, word_nums, hash_tokens_nums
        self.label = tf.placeholder(tf.float32, shape=[batch_size])

        self.predict_op = self.create_model_op()
        self.loss_op = self.create_loss_op()
        self.train_op = self.create_train_op()


    def create_model_op(self):

        '''
            建立整个模型，分成两端的网络，query端和doc端的
        '''

        features = []
        structures = ['query_dnn', 'doc_dnn']
        input_dict = {
            structures[0]: self.input_q,
            structures[1]: self.input_doc
        }

        for one_structrue in structures:
            with tf.variable_scope(one_structrue):
                node_nums = [self.hash_token_nums] + \
                            [self.dnn_hidden_node_nums] * self.dnn_layer_nums + [self.feature_nums]
                result = input_dict[one_structrue]
                for i in range(len(node_nums)-1):
                    w = tf.Variable(
                        tf.random_uniform([node_nums[i], node_nums[i+1]], -0.001, 0.001), name='weights'+str(i)
                    )
                    # 网络比较深，参数比较多时，注意w取值应该比较小，学习率适当增大
                    b = tf.Variable(tf.zeros([node_nums[i+1]]), name="bias"+str(i))
                    result = tf.matmul(result, w) + b
                    result = tf.nn.sigmoid(result)
                    features.append(result)

        self.predict_query = features[0]
        self.predict_doc = features[1]
        norms1 = tf.sqrt(tf.reduce_sum(tf.square(features[0]), 1, keep_dims=False))
        norms2 = tf.sqrt(tf.reduce_sum(tf.square(features[1]), 1, keep_dims=False))
        self.relevance = tf.reduce_sum(features[0] * features[1], 1) / norms1 / norms2
        return self.relevance


    def create_loss_op(self):
        '''
            计算相关文档的loss, gama经验值也用来学习
        :return:
        '''
        gama = tf.Variable(tf.random_uniform([1]), name="gama")
        ret = self.predict_op * gama
        ret = tf.reshape(ret, [-1, self.neg_nums+1])
        ret = tf.log(tf.nn.softmax(ret))
        ret = tf.reduce_sum(ret, 0)
        return -tf.gather(ret, 0)


    def create_train_op(self):
        '''
            采用梯度下降方式学习
        :return:
        '''
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_op)


    def creat_feed_dict(self, query_batch, doc_batch, label_batch):
        '''
        :param query_batch: 查询输入
        :param doc_batch: 文档输入
        :param label_batch: 查询和文档对应的相关性label
        :return:
        '''
        self.feed_dict = {
            self.input_q : query_batch,
            self.input_doc: doc_batch,
            self.label : label_batch
        }


    def run_epoch(self, sess, query_input, doc_input, labels):
        '''
        计算一次迭代过程
        :param sess:
        :param query_input:
        :param doc_input:
        :param labels:
        :return:
        '''
        average_loss = 0
        for step, (query, doc, label) in enumerate(
                utils.data_iterator(query_input, doc_input, labels, self.batch_size)
            ):
            # print query[1, 1], doc[1, 1], label[1]
            self.creat_feed_dict(query, doc, label)
            _, loss_value, predict_query, predict_doc, relevance = sess.run([self.train_op, self.loss_op, self.predict_query\
                    , self.predict_doc, self.relevance], feed_dict=self.feed_dict)
            average_loss += loss_value
            #print 'step ', step, loss_value
            #print 'predict ', predict_query[0], predict_doc[0], relevance[0]
        return average_loss / step


    def fit(self, sess, query_input, doc_input, labels):
        '''
        模型入口
        :param sess:
        :param query_input:
        :param doc_input:
        :param labels:
        :return:
        '''
        losses = []
        for epoch in range(self.max_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, query_input, doc_input, labels)
            duration = time.time() - start_time

            print('Epoch %d: loss = %.5f (%.3f sec)'
             % (epoch, average_loss, duration))
            losses.append(average_loss)
        return losses

    def predict(self, sess, query, doc, labels):
        '''
        计算预测过后的查询与文档的相关性
        :param sess:
        :param query:
        :param doc:
        :param labels:
        :return:
        '''
        self.creat_feed_dict(query, doc, labels)
        predict = sess.run(self.relevance, feed_dict=self.feed_dict)


def test_dssm():
    '''
    测试函数
    :return:
    '''
    with tf.Graph().as_default():
        tf.set_random_seed(1)
        model = DSSM(hash_tokens_nums=30000, dnn_layer_nums=2, dnn_hidden_node_nums=300, feature_nums=128,
                batch_size=10, neg_nums=4, learning_rate=0.02, max_epochs=500)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        np.random.seed(1)
        query = np.random.rand(500, 30000)
        doc = np.random.rand(500, 30000)
        label = np.array([1, 0, 0, 0, 0] * 100)

        #print query
        #print doc
        #print label

        losses = model.fit(sess, query, doc, label)

    print losses[-1]


if __name__ == '__main__':
    test_dssm()
