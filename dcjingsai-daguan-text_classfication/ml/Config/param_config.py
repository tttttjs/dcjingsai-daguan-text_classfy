
"""
__file__

    param_config.py

__description__

    This file provides global parameter configurations for the project.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import os
import numpy as np


############
## Config ##
############
class ParamConfig:
    def __init__(self, feat_folder):
    
        self.seed = 2018
        self.n_classes = 19

        ## CV params
        #self.n_runs = 3
        self.n_folds = 3
        self.stratified_label = "class"

        ## path
        self.data_folder = "../Input"
        self.feat_folder = feat_folder
        self.original_train_data_path = "%s/train_set.csv" % self.data_folder
        self.original_test_data_path = "%s/test_set.csv" % self.data_folder
        self.processed_train_data_path = "%s/train_set.processed.csv.pkl" % self.data_folder
        self.processed_test_data_path = "%s/test_set.processed.csv.pkl" % self.data_folder
        #self.pos_tagged_train_data_path = "%s/train.pos_tagged.csv.pkl" % self.feat_folder
        #self.pos_tagged_test_data_path = "%s/test.pos_tagged.csv.pkl" % self.feat_folder

        self.skf = '{0}/stratifiedKFold.{1}.pkl'.format(self.data_folder, self.stratified_label)
        self.train_labels = '{0}/train_labels.pkl'.format(self.data_folder)
        self.test_ids = '{0}/test_ids.pkl'.format(self.data_folder)

        self.train_char_cnt = '{0}/train.char_cnt.feat.pkl'.format(self.feat_folder)
        self.train_word_cnt = '{0}/train.word_cnt.feat.pkl'.format(self.feat_folder)
        self.test_char_cnt = '{0}/train.char_cnt.feat.pkl'.format(self.feat_folder)
        self.test_word_cnt = '{0}/test.word_cnt.feat.pkl'.format(self.feat_folder)

        ### nlp related        
        #self.drop_html_flag = drop_html_flag
        #self.basic_tfidf_ngram_range = basic_tfidf_ngram_range
        #self.basic_tfidf_vocabulary_type = basic_tfidf_vocabulary_type
        #self.cooccurrence_tfidf_ngram_range = cooccurrence_tfidf_ngram_range
        #self.cooccurrence_word_exclude_stopword = cooccurrence_word_exclude_stopword
        #self.stemmer_type = stemmer_type

        ### transform for count features
        #self.count_feat_transform = count_feat_transform

        ### create feat folder
        #if not os.path.exists(self.feat_folder):
        #    os.makedirs(self.feat_folder)

        ### creat folder for the training and testing feat
        #if not os.path.exists("%s/All" % self.feat_folder):
        #    os.makedirs("%s/All" % self.feat_folder)

        ## creat folder for each run and fold
        for fold in range(1,self.n_folds+1):
            path = "%s/Fold%d" % (self.feat_folder, fold)
            if not os.path.exists(path):
                os.makedirs(path)


## initialize a param config					
config = ParamConfig(feat_folder="../Feat")
