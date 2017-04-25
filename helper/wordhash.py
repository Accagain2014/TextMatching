#coding=utf-8
import numpy as np
import pickle
import distance
from scipy import sparse as sps

class WordHash(object):

    '''
        Implement word hash methods mentioned in the paper: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    '''

    def __init__(self, words, n_gram=3, marks='#', load_from_file=False, dump_to_file=False, file=None):
        '''

        :param words: origin vacabulary
        :param n_gram: number of letters to make a n_gram term
        :param marks: the character added in the starting and ending position of a word
        :param load_from_file: load n_gram index map dict from file or not
        :param load_file: load file name
        '''

        if load_from_file:
            with open(file, 'rb') as fr:
                self.__dict__ = pickle.load(fr).__dict__  # load an object
            return

        self.ori_words = words
        self.ori_len = len(self.ori_words)
        self.words = map(lambda x: marks+x.lower()+marks, list(set(self.ori_words)))
        self.ori_diff_len = len(self.words)
        self.n_gram = n_gram
        self.marks = marks
        self.hashed_words = set()
        self.n_gram_index_map = {}
        self.n_gram_size = 0

        print 'Sum number of origin words: ', self.ori_len
        print 'Sum number of origin diff words: ', self.ori_diff_len
        print 'Letter n-gram: ', self.n_gram

        self._get_hash_dict()

        if dump_to_file:
            with open(file, 'wb') as fw:
                pickle.dump(self, fw)


    def _split(self, word):
        '''
            Split a word with lenth of self.n_gram_size
        :param self:
        :param word: word to be splited by n_gram len
        :return:
        '''

        splited_ngrams = []
        word_len = len(word)
        split_point = 0
        while split_point < word_len-1: # don't consider the last marks
            splited_ngrams.append(word[split_point : split_point+self.n_gram])
            split_point += 1
        return splited_ngrams


    def _get_hash_dict(self):
        '''
            Get n_gram terms and mapping them to indexes
        :return:
        '''

        for one_word in self.words:
            ngram_words = self._split(one_word)
            self.hashed_words = self.hashed_words | set(ngram_words)

        word_keys = list(self.hashed_words)
        word_values = range(0, len(word_keys))
        self.n_gram_index_map = dict(zip(word_keys, word_values))
        self.n_gram_size = len(word_keys)

        print 'Sum numbers of n-grams: ', self.n_gram_size
        return self.hashed_words

    def get_n_gram_count(self, sentences, is_dump=False, dump_file=None):
        '''
            Get n_gram counting term matrix
        :param sentences: sentences to be handled to get n_gram term counting matrix
        :param is_dump: whether dump the result to file or not
        :param dump_file: dump file name
        :return: n_gram term counting sparse matrix, shapes(sentences number, n_gram term size)
        '''

        # n_gram_count = np.zeros((len(sentences), self.n_gram_size))
        n_gram_count = sps.lil_matrix((len(sentences), self.n_gram_size))
        sen_cnt = 0
        for one_sen in sentences:
            one_sen = one_sen.strip().split()
            for one_word in one_sen:
                one_word = one_word.strip()
                one_word = self.marks+one_word.lower()+self.marks
                splited_n_gram = self._split(one_word)
                n_gram_index = map(lambda x: self.n_gram_index_map[x], splited_n_gram)
                # n_gram_count[sen_cnt, n_gram_index] += 1
                for one_n_gram_index in n_gram_index:
                    n_gram_count[sen_cnt, one_n_gram_index] += 1
            sen_cnt += 1
        if is_dump:
            with open(dump_file, 'wb') as fw:
                pickle.dump(n_gram_count.tocsr(), fw)
            print 'Dump to file ', dump_file, ' done.'
        print 'Get n_gram count matrix done, shape with: ', n_gram_count.shape
        return n_gram_count.tocsr()

def test_WordHash():

    sentence = 'Key words based text matching methods and semantic matching methods'

    print sentence.split()
    wordhash = WordHash(sentence.split(), load_from_file=False, load_file='n_gram_term_index_mapping.pkl', dump_to_file=False, dump_file='n_gram_term_index_mapping.pkl')
    print wordhash.n_gram_index_map
    n_gram_matrix = wordhash.get_n_gram_count(['key words text matching methods', 'semantic text matching methods'])

    print distance.cos_dis(n_gram_matrix[0].toarray().reshape([-1]), n_gram_matrix[1].toarray().reshape([-1]))

if __name__ == '__main__':
    test_WordHash()