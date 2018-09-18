#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
######################################################################## 
'''
File: train.py
Author: work(work@baidu.com)
Date: 2017/05/08 15:01:13
'''

import os
import sys
import timeit
import random
import argparse
import logging
from os import getcwd
from os.path import join
from math import sqrt

import cPickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.append('../Config')
from param_config import config

def parse_args():
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser()
    parser.add_argument('step')
    args = vars(parser.parse_args())

    return args


def parse_step(step, num_steps):
    if step == 'all':
        res = range(num_steps)
    elif '.' in step:
        res = [step]
    elif '-' not in step:
        res = [int(step)]
    else:
        step_be = step.split('-')
        if step_be[1]:
            res = range(int(step_be[0]), int(step_be[1])+1)
        else:
            res = range(int(step_be[0]), num_steps)
    return res


def load_data(logger):
    train_data_file = '../Input/train_set.csv'
    train = pd.read_csv(train_data_file, sep=',')
    logger.info('trainfile: ' + train_data_file)
    print 'train.shape:', train.shape

    test_data_file = '../Input/test_set.csv'
    test = pd.read_csv(test_data_file, sep=',')
    logger.info('testfile: ' + test_data_file)
    print 'test.shape:', test.shape

    return train, test


def generate_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('train.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

#def main():
args = parse_args()
logger = generate_logger()

num_steps = 3
step_list = parse_step(args['step'], num_steps)

if 0 in step_list:
    seed = config.seed
    print 'seed:', seed
    np.random.seed(seed)
    train, test = load_data(logger)

    train['class'] = train['class'] - 1

    columns = list(train.columns)
    feature_names = list(columns)
    print 'len(features):', len(feature_names)
    print 'features:', str(feature_names)

    feature_name = 'word_seg'

    with open('{0}/stratifiedKFold.{1}.pkl'.format(config.data_folder, \
            config.stratified_label), 'rb') as f:
        skf = cPickle.load(f)

    for i, (train_index, test_index) in enumerate(skf, 1):
        print 'Running Fold', i, '/', config.n_folds
        train_off = train.iloc[train_index]
        test_off = train.iloc[test_index]
        print train_off.shape, test_off.shape

        path = '{0}/Fold{1}'.format(config.feat_folder, i)
        vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
        train_off_term_doc = vec.fit_transform(train_off[feature_name])
        print train_off_term_doc.shape

        train_off_feat_file = '{}/train_off.basic_tfidf.feat.pkl'.format(path)
        if not os.path.isfile(train_off_feat_file):
            with open(train_off_feat_file, 'wb') as f:
                cPickle.dump(train_off_term_doc, f)

        test_off_term_doc = vec.transform(test_off[feature_name])
        print test_off_term_doc.shape

        test_off_feat_file = '{}/test_off.basic_tfidf.feat.pkl'.format(path)
        if not os.path.isfile(test_off_feat_file):
            with open(test_off_feat_file, 'wb') as f:
                cPickle.dump(test_off_term_doc, f)

    #vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
    #train_term_doc = vec.fit_transform(train[feature_name])
    #print train_term_doc.shape

    #test_term_doc = vec.transform(test[feature_name])
    #print test_term_doc.shape

    #train_feat_file =  '{0}/train.basic_tfidf.feat.pkl'.format(config.feat_folder)
    #if not os.path.isfile(train_feat_file):
    #    with open(train_feat_file, 'wb') as f:
    #        cPickle.dump(train_term_doc, f)

    #test_feat_file = '{0}/test.basic_tfidf.feat.pkl'.format(config.feat_folder)
    #if not os.path.isfile(test_feat_file):
    #    with open(test_feat_file, 'wb') as f:
    #        cPickle.dump(test_term_doc, f)


#if __name__ == '__main__':
#    main()
