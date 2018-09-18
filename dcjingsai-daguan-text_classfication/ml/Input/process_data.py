#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 process_data.py
@author: liaolin (liaolin@baidu.com)
@date:	 2018-07-11 09:24:09
@brief:	  
"""
import sys

import _pickle as cPickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append('../Config')
from param_config import config

def main():
    df_train = pd.read_csv(config.original_train_data_path, sep=',')
    df_train['class'] = df_train['class'] - 1
    #with open(config.processed_train_data_path, 'wb') as f:
    #    cPickle.dump(df_train, f)

    df_test = pd.read_csv(config.original_test_data_path, sep=',')
    #with open(config.processed_test_data_path, 'wb') as f:
    #    cPickle.dump(df_test, f)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    stratified_label = config.stratified_label
    list_skf = list(skf.split(df_train.values, df_train[stratified_label].values))
    with open('{0}/stratifiedKFold.{1}.pkl'.format(config.data_folder, stratified_label), \
            'wb') as f:
        cPickle.dump(list_skf, f)

    with open('{0}/train_labels.pkl'.format(config.data_folder), 'wb') as f:
        cPickle.dump(df_train['class'].values, f)

    with open('{0}/test_ids.pkl'.format(config.data_folder), 'wb') as f:
        cPickle.dump(df_test['id'].values, f)


if __name__ == '__main__':
    main()
