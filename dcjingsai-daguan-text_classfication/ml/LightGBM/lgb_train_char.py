# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import argparse
import _pickle as cPickle

import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score

sys.path.append('../Config')
from param_config import config

def eval_score(preds, train_data):
    labels = train_data.get_label()
    preds = preds.reshape(config.n_classes, -1)
    preds = np.argmax(preds, axis=0)
    return ('f1-score', f1_score(labels, preds, average='macro'), True)

with open(config.skf, 'rb') as f:
    skf = cPickle.load(f)

train_labels_file = config.train_labels
with open(train_labels_file, 'rb') as f:
    train_labels = cPickle.load(f)

feat_name = 'char_basic_tfidf'
for i in range(1, 2):
    train_off_feat_file = '{0}/Fold{1}/train_off.{2}.feat.pkl'.format(config.feat_folder, \
            i, feat_name)
    test_off_feat_file = '{0}/Fold{1}/test_off.{2}.feat.pkl'.format(config.feat_folder, \
            i, feat_name)

    print('load feature')
    with open(train_off_feat_file, 'rb') as f:
        train_off_term_doc = cPickle.load(f)

    with open(test_off_feat_file, 'rb') as f:
        test_off_term_doc = cPickle.load(f)

    train_idx, valid_idx = skf[i - 1]
    train_off_labels = train_labels[train_idx]
    test_off_labels = train_labels[valid_idx]

    lgb_train = lgb.Dataset(train_off_term_doc, train_off_labels)
    lgb_eval = lgb.Dataset(test_off_term_doc, test_off_labels, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': config.n_classes,
        'metric_freq': 1,
        'max_bin': 255,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'metric': 'None'
    }

    print('begin train')
    num_boost_round = 3000
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    feval=eval_score)

    print('best_iteration:', gbm.best_iteration)
    preds = gbm.predict(test_off_term_doc)

    test_preds_file = '../Ensemble/Fold1/lgb_char_preds_off.pkl'
    with open(test_preds_file, 'wb') as f:
        cPickle.dump(preds, f)
