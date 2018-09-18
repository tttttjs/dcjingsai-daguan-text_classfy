# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import argparse
import pickle
import _pickle as cPickle

import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score

sys.path.append('../Config')
from param_config import config

train_labels_file = config.train_labels
feat_name = 'char_basic_tfidf'
train_on_feat_file = '{0}/train.{1}.feat.pkl'.format(config.feat_folder, feat_name)
test_on_feat_file = '{0}/test.{1}.feat.pkl'.format(config.feat_folder, feat_name)

with open(train_labels_file, 'rb') as f:
    train_labels = pickle.load(f)

with open(train_on_feat_file, 'rb') as f:
    train_on_term_doc = pickle.load(f)

with open(test_on_feat_file, 'rb') as f:
    test_on_term_doc = pickle.load(f)

print('train.shape:', train_on_term_doc.shape)
print('test.shape:', test_on_term_doc.shape)
lgb_train = lgb.Dataset(train_on_term_doc, train_labels)

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
}

print('begin train')
num_boost_round = 741
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_train,
                num_boost_round=num_boost_round)

preds = gbm.predict(test_on_term_doc)
test_preds_file = '../Ensemble/Preds_On/lgb_char_preds_on.pkl'
with open(test_preds_file, 'wb') as f:
    cPickle.dump(preds, f)
