#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 lda_train.py
@author: liaolin (liaolin@baidu.com)
@date:	 2018-08-22 21:11:27
@brief:	 train lda model
"""
import sys
import argparse
import pickle
import _pickle as cPickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

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

args = parse_args()

num_steps = 2
step_list = parse_step(args['step'], num_steps)

with open(config.skf, 'rb') as f:
    skf = cPickle.load(f)

feature_name = 'word_seg'

if 0 in step_list:
    for i in range(1, 2):
        train_idx, valid_idx = skf[i - 1]

        train_off_tf_file = './train_off_tf.fold{}.pkl'.format(i)
        with open(train_off_tf_file, 'rb') as f:
            train_off_tf = cPickle.load(f)

        test_off_tf_file = './test_off_tf.fold{}.pkl'.format(i)
        with open(test_off_tf_file, 'rb') as f:
            test_off_tf = cPickle.load(f)

        print('train lda')
        n_components = 100
        learning_method = 'online'
        batch_size = 1000
        n_jobs = 5
        verbose = 3
        lda = LDA(n_components=n_components,
                  learning_method=learning_method,
                  batch_size=batch_size,
                  verbose=verbose,
                  n_jobs=n_jobs,
                  random_state=2018)

        train_off_topics = lda.fit_transform(train_off_tf)
        test_off_topics = lda.transform(test_off_tf)
        train_off_lda_file = './train_off_lda.fold{}.pkl'.format(i)
        with open(train_off_lda_file, 'wb') as f:
            cPickle.dump(train_off_topics, f)

        test_off_lda_file = './test_off_lda.fold{}.pkl'.format(i)
        with open(test_off_lda_file, 'wb') as f:
            cPickle.dump(test_off_topics, f)

if 1 in step_list:
    train_on_tf_file = './train_on_tf.pkl'
    with open(train_on_tf_file, 'rb') as f:
        train_on_tf = cPickle.load(f)

    test_on_tf_file = './test_on_tf.pkl'
    with open(test_on_tf_file, 'rb') as f:
        test_on_tf = cPickle.load(f)

    print('train lda')
    n_components = 100
    learning_method = 'online'
    batch_size = 1000
    n_jobs = 5
    verbose = 3
    lda = LDA(n_components=n_components,
              learning_method=learning_method,
              batch_size=batch_size,
              verbose=verbose,
              n_jobs=n_jobs,
              random_state=2018)

    train_on_topics = lda.fit_transform(train_on_tf)
    test_on_topics = lda.transform(test_on_tf)
    train_on_lda_file = './train_on_lda.pkl'
    with open(train_on_lda_file, 'wb') as f:
        cPickle.dump(train_on_topics, f)

    test_on_lda_file = './test_on_lda.pkl'
    with open(test_on_lda_file, 'wb') as f:
        cPickle.dump(test_on_topics, f)
