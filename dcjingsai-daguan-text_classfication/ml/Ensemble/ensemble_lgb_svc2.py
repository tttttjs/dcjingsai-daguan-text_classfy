#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 recover_ensemble_coef.py
@author: liaolin (liaolin@baidu.com)
@date:	 2018年09月04日 星期二 14时07分12秒
@brief:	  
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import _pickle as cPickle
from sklearn.metrics import f1_score

sys.path.append('../Config')
from param_config import config

def parse_args():
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser()
    parser.add_argument('step')
    parser.add_argument('sub_id')
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

if 0 in step_list:
    fold = 1
    train_idx, valid_idx = skf[fold - 1]
    test_off_labels = train_labels[valid_idx]

    lgb_word_preds_off_file = './Fold{}/lgb_word_lda_embed_preds_off.pkl'.format(fold)
    with open(lgb_word_preds_off_file, 'rb') as f:
        lgb_word_preds_off = cPickle.load(f)
    print(lgb_word_preds_off)
    print('lgb_word:', eval_f1_score(test_off_labels, lgb_word_preds_off))

    lgb_char_preds_off_file = './Fold{}/lgb_char_preds_off.pkl'.format(fold)
    with open(lgb_char_preds_off_file, 'rb') as f:
        lgb_char_preds_off = cPickle.load(f)
    print('lgb_char:', eval_f1_score(test_off_labels, lgb_char_preds_off))

    svc_word_preds_off_file = './Fold{}/svc_word_preds_off.pkl'.format(fold)
    with open(svc_word_preds_off_file, 'rb') as f:
        svc_word_preds_off = cPickle.load(f)
    print('svc_word:', eval_f1_score(test_off_labels, svc_word_preds_off))

    svc_char_preds_off_file = './Fold{}/svc_char_preds_off.pkl'.format(fold)
    with open(svc_char_preds_off_file, 'rb') as f:
        svc_char_preds_off = cPickle.load(f)
    print('svc_char:', eval_f1_score(test_off_labels, svc_char_preds_off))

    ensemble_score_dict = {}
    for coef1 in [a / 20.0 for a in range(20)]:
        res1 = 20 - int(coef1 * 20.0)
        for coef2 in [b / 20.0 for b in range(res1)]:
            res2 = 20 - int(coef1 * 20.0) - int(coef2 * 20.0)
            for coef3 in [c / 20.0 for c in range(res2)]:
                coef4 = 1 - coef1 - coef2 - coef3
                coef_pair = (coef1, coef2, coef3, coef4)
                ensemble_preds = coef1 * lgb_word_preds_off + coef2 * lgb_char_preds_off + coef3 * svc_word_preds_off + coef4 * svc_char_preds_off
                preds_score = eval_f1_score(test_off_labels, ensemble_preds)
                ensemble_score_dict[coef_pair] = preds_score
    sorted_ensemble_score_items = sorted(ensemble_score_dict.items(), key=lambda x:x[1], reverse=True)
    print('sorted_ensemble_score_items:', sorted_ensemble_score_items[:5])


if 1 in step_list:
    lgb_word_preds_on_file = './Preds_On/lgb_word_lda_embed_preds_on.pkl'
    with open(lgb_word_preds_on_file, 'rb') as f:
        lgb_word_preds_on = cPickle.load(f)

    lgb_char_preds_on_file = './Preds_On/lgb_char_preds_on.pkl'
    with open(lgb_char_preds_on_file, 'rb') as f:
        lgb_char_preds_on = cPickle.load(f)

    svc_word_preds_on_file = './Preds_On/svc_word_preds_on.pkl'
    with open(svc_word_preds_on_file, 'rb') as f:
        svc_word_preds_on = cPickle.load(f)

    svc_char_preds_on_file = './Preds_On/svc_char_preds_on.pkl'
    with open(svc_char_preds_on_file, 'rb') as f:
        svc_char_preds_on = cPickle.load(f)

    sub_id = args['sub_id']

    #coef1 = 0.2
    #coef2 = 0.15
    #coef3 = 0.15
    #coef4 = 0.5
    #coef1 = 0.3
    #coef2 = 0.2
    #coef3 = 0.35
    #coef4 = 0.15
    #coef1 = 0.2
    #coef2 = 0.1
    #coef3 = 0.3
    #coef4 = 0.4
    #coef1 = 0.35
    #coef2 = 0.15
    #coef3 = 0.35
    #coef4 = 0.15
    coef1 = 0.3
    coef2 = 0.2
    coef3 = 0.3
    coef4 = 0.2

    ensemble_preds_on = coef1 * lgb_word_preds_on + coef2 * lgb_char_preds_on + coef3 * svc_word_preds_on + coef4 * svc_char_preds_on
    ensemble_preds_on_file = './Preds_On/ensemble_lgb_svc_preds{}'.format(sub_id)
    np.save(ensemble_preds_on_file, ensemble_preds_on)

    ensemble_labels_on = np.argmax(ensemble_preds_on, axis=1) + 1
    test_ids_file = '../Input/test_ids.pkl'
    with open(test_ids_file, 'rb') as f:
        test_ids = cPickle.load(f)
    test_preds = pd.DataFrame({'id':test_ids, 'class':ensemble_labels_on})
    sub_names = ['id', 'class']
    test_preds[sub_names].to_csv('../Submit/ensemble_lgb_svc{}.csv'.format(sub_id), index=False)
