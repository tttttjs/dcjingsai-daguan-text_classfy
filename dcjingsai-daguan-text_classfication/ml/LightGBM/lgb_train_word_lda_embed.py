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

def eval_score(preds, train_data):
    labels = train_data.get_label()
    preds = preds.reshape(config.n_classes, -1)
    preds = np.argmax(preds, axis=0)
    return ('f1-score', f1_score(labels, preds, average='macro'), True)

args = parse_args()

num_steps = 2
step_list = parse_step(args['step'], num_steps)

with open(config.skf, 'rb') as f:
    skf = cPickle.load(f)

train_labels_file = config.train_labels
with open(train_labels_file, 'rb') as f:
    train_labels = cPickle.load(f)

feat_name = 'word_basic_tfidf'

embed_fea_file = '../Feat/average_embed_fea.pkl'
with open(embed_fea_file, 'rb') as f:
    embed_fea = cPickle.load(f)

s_embed_fea = sparse.csr_matrix(embed_fea)

if 0 in step_list:
    for i in range(1, 2):
        train_off_tfidf_file = '{0}/Fold{1}/train_off.{2}.feat.pkl'.format(config.feat_folder, \
                i, feat_name)
        test_off_tfidf_file = '{0}/Fold{1}/test_off.{2}.feat.pkl'.format(config.feat_folder, \
                i, feat_name)

        print('load feature')
        with open(train_off_tfidf_file, 'rb') as f:
            train_off_tfidf = pickle.load(f)

        with open(test_off_tfidf_file, 'rb') as f:
            test_off_tfidf = pickle.load(f)

        train_off_lda_file = '../LDA/train_off_lda.fold{}.pkl'.format(i)
        test_off_lda_file = '../LDA/test_off_lda.fold{}.pkl'.format(i)
        with open(train_off_lda_file, 'rb') as f:
            train_off_lda = cPickle.load(f)
        print('train_off_lda.shape:', train_off_lda.shape)

        with open(test_off_lda_file, 'rb') as f:
            test_off_lda = cPickle.load(f)
        print('test_off_lda.shape:', test_off_lda.shape)

        train_idx, valid_idx = skf[i - 1]
        train_off_labels = train_labels[train_idx]
        test_off_labels = train_labels[valid_idx]

        s_train_off_lda = sparse.csr_matrix(train_off_lda)
        s_test_off_lda = sparse.csr_matrix(test_off_lda)

        s_train_off_embed = s_embed_fea[train_idx]
        s_test_off_embed = s_embed_fea[valid_idx]

        train_off_feat = sparse.hstack([train_off_tfidf, s_train_off_lda, s_train_off_embed], format='csr')
        test_off_feat = sparse.hstack([test_off_tfidf, s_test_off_lda, s_test_off_embed], format='csr')

        lgb_train = lgb.Dataset(train_off_feat, train_off_labels)
        lgb_eval = lgb.Dataset(test_off_feat, test_off_labels, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': config.n_classes,
            'metric_freq': 1,
            'max_bin': 255,
            'num_leaves': 41,
            'learning_rate': 0.05,
            'metric': 'None'
        }
        #'num_leaves': 31,

        print('begin train')
        num_boost_round = 3000
        #num_boost_round = 540
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=lgb_eval,
                        feval=eval_score)
                        #early_stopping_rounds=100,

        print('best_iteration:', gbm.best_iteration)
        preds = gbm.predict(test_off_feat)

        #test_preds_file = '../Ensemble/Fold1/lgb_word_lda_preds_off2.pkl'
        test_preds_file = '../Ensemble/Fold1/lgb_word_lda_preds_off3.pkl'
        with open(test_preds_file, 'wb') as f:
            cPickle.dump(preds, f)

        best_iteration = gbm.best_iteration


if 1 in step_list:
    num_train = 102277

    train_on_tfidf_file = '{0}/train.{1}.feat.pkl'.format(config.feat_folder, feat_name)
    test_on_tfidf_file = '{0}/test.{1}.feat.pkl'.format(config.feat_folder, feat_name)
    with open(train_on_tfidf_file, 'rb') as f:
        train_on_tfidf = pickle.load(f)

    with open(test_on_tfidf_file, 'rb') as f:
        test_on_tfidf = pickle.load(f)

    train_on_lda_file = '../LDA/train_on_lda.pkl'
    test_on_lda_file = '../LDA/test_on_lda.pkl'
    with open(train_on_lda_file, 'rb') as f:
        train_on_lda = cPickle.load(f)

    with open(test_on_lda_file, 'rb') as f:
        test_on_lda = cPickle.load(f)

    s_train_on_lda = sparse.csr_matrix(train_on_lda)
    s_test_on_lda = sparse.csr_matrix(test_on_lda)

    s_train_on_embed = s_embed_fea[:num_train]
    s_test_on_embed = s_embed_fea[num_train:]

    train_on_feat = sparse.hstack([train_on_tfidf, s_train_on_lda, s_train_on_embed], format='csr')
    test_on_feat = sparse.hstack([test_on_tfidf, s_test_on_lda, s_test_on_embed], format='csr')

    lgb_train = lgb.Dataset(train_on_feat, train_labels)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': config.n_classes,
        'metric_freq': 1,
        'max_bin': 255,
        'num_leaves': 41,
        'learning_rate': 0.05,
    }
    #'num_leaves': 31,

    print('begin train')
    #num_boost_round = 636
    #num_boost_round = 540
    num_boost_round = 662
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_train,
                    num_boost_round=num_boost_round)

    preds = gbm.predict(test_on_feat)
    test_preds_file = '../Ensemble/Preds_On/lgb_word_lda_embed_preds_on.pkl'
    with open(test_preds_file, 'wb') as f:
        cPickle.dump(preds, f)
