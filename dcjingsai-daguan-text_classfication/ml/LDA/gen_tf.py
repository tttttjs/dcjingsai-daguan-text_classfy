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


df_train = pd.read_csv('../Input/train_set.csv', sep=',')
df_test = pd.read_csv('../Input/test_set.csv', sep=',')

num_train = df_train.shape[0]
num_test = df_test.shape[0]

df = pd.concat([df_train, df_test], ignore_index=True)

args = parse_args()

num_steps = 2
step_list = parse_step(args['step'], num_steps)

feature_name = 'word_seg'

if 0 in step_list:
    for i in range(1, 2):
        train_idx, valid_idx = skf[i - 1]

        train_off_texts = df.iloc[train_idx]['word_seg'].values
        test_off_texts = df.iloc[valid_idx]['word_seg'].values

        vec = CountVectorizer(min_df=3, max_df=0.9)
        train_off_tf = vec.fit_transform(train_off_texts)
        test_off_tf = vec.transform(test_off_texts)

        train_off_tf_file = './train_off_tf.fold{}.pkl'.format(i)
        with open(train_off_tf_file, 'wb') as f:
            cPickle.dump(train_off_tf, f)

        test_off_tf_file = './test_off_tf.fold{}.pkl'.format(i)
        with open(test_off_tf_file, 'wb') as f:
            cPickle.dump(test_off_tf, f)

if 1 in step_list:
    train_on_texts = df.iloc[:num_train]['word_seg'].values
    test_on_texts = df.iloc[num_train:]['word_seg'].values

    vec = CountVectorizer(min_df=3, max_df=0.9)
    train_on_tf = vec.fit_transform(train_on_texts)
    test_on_tf = vec.transform(test_on_texts)

    train_on_tf_file = './train_on_tf.pkl'
    with open(train_on_tf_file, 'wb') as f:
        cPickle.dump(train_on_tf, f)

    test_on_tf_file = './test_on_tf.pkl'
    with open(test_on_tf_file, 'wb') as f:
        cPickle.dump(test_on_tf, f)
