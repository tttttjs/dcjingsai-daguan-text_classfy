#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 NBSVM_train.py
@author: liaolin (liaolin@baidu.com)
@date:	 2018-08-06 21:32:12
@brief:	  
"""
import sys
import argparse
import pickle
import _pickle as cPickle

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_fscore_support, f1_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy import sparse

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

def eval_f1_score(labels, preds):
    label_preds = np.argmax(preds, axis=1)
    return f1_score(labels, label_preds, average='macro')

args = parse_args()

num_steps = 2
step_list = parse_step(args['step'], num_steps)

with open(config.skf, 'rb') as f:
    skf = cPickle.load(f)

train_labels_file = config.train_labels
with open(train_labels_file, 'rb') as f:
    train_labels = cPickle.load(f)

feat_name = 'word_tfidf'
if 0 in step_list:
    eval_score_list = []
    for i in xrange(1, 4):
        print('FOLD {}/3'.format(i))

        train_off_feat_file = '{0}/Fold{1}/train_off.{2}.feat.pkl'.format(config.feat_folder, \
                i, feat_name)
        test_off_feat_file = '{0}/Fold{1}/test_off.{2}.feat.pkl'.format(config.feat_folder, \
                i, feat_name)

        with open(train_off_feat_file, 'rb') as f:
            train_off_term_doc = cPickle.load(f)

        with open(test_off_feat_file, 'rb') as f:
            test_off_term_doc = cPickle.load(f)

        train_off_feat = train_off_term_doc
        test_off_feat = test_off_term_doc

        train_idx, valid_idx = skf[i - 1]

        train_off_labels = train_labels[train_idx]
        test_off_labels = train_labels[valid_idx]

        clf = CalibratedClassifierCV(base_estimator=LinearSVC(C=0.9, class_weight='balanced'))
        clf.fit(train_off_feat, train_off_labels)
        preds = clf.predict_proba(test_off_feat)

        test_off_preds_file = '../Ensemble/Fold{}/svc_word_preds_off.pkl'.format(i)
        with open(test_off_preds_file, 'wb') as f:
            cPickle.dump(preds, f)

        eval_score = eval_f1_score(test_off_labels, preds)
        print('f1_score:', eval_score)

        eval_score_list.append(eval_score)

    eval_score_array = np.array(eval_score_list)
    print('mean f1_score:', np.mean(eval_score_list))


if 1 in step_list:
    train_on_feat_file = '{0}/train.{1}.feat.pkl'.format(config.feat_folder, feat_name)
    test_on_feat_file = '{0}/test.{1}.feat.pkl'.format(config.feat_folder, feat_name)

    with open(train_on_feat_file, 'rb') as f:
        train_on_term_doc = cPickle.load(f)

    with open(test_on_feat_file, 'rb') as f:
        test_on_term_doc = cPickle.load(f)

    print('train_on_term_doc.shape:', train_on_term_doc.shape)
    print('test_on_term_doc.shape:', test_on_term_doc.shape)

    clf = CalibratedClassifierCV(base_estimator=LinearSVC(C=0.9, class_weight='balanced'))
    clf.fit(train_on_term_doc, train_labels)
    preds = clf.predict_proba(test_on_term_doc)

    test_on_preds_file = '../Ensemble/Preds_On/svc_word_preds_on.pkl'
    with open(test_on_preds_file, 'wb') as f:
        cPickle.dump(preds, f)
