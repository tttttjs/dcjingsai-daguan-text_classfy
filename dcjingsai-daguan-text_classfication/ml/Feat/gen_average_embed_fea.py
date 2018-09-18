#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 gen_average_embed_fea.py
@author: liaolin (liaolin@baidu.com)
@date:	 2018-08-25 10:53:34
@brief:	  
"""
import sys
import _pickle as cPickle
sys.path.append('../Config')
from param_config import config

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

word_vectors_file = '../Feat/word_vectors_size300'
wv = KeyedVectors.load(word_vectors_file, mmap='r')

word2vec_dict = {k: wv[k] for k, v in wv.vocab.items()}

df_train = pd.read_csv('../Input/train_set.csv', sep=',')
df_test = pd.read_csv('../Input/test_set.csv', sep=',')
df = pd.concat([df_train, df_test], ignore_index=True)
df['word_list'] = df['word_seg'].apply(lambda x: [w for w in x.split(' ')])

all_word_list = df['word_list'].values

average_embed_fea_list = []
for i, word_list in enumerate(all_word_list):
    embed_array = np.array([word2vec_dict[word] for word in word_list if word in word2vec_dict])
    average_embed_fea = np.mean(embed_array, axis=0)
    average_embed_fea_list.append(average_embed_fea)

average_embed_fea_array = np.array(average_embed_fea_list)

average_embed_fea_pkl = './average_embed_fea.pkl'
with open(average_embed_fea_pkl, 'wb') as f:
    cPickle.dump(average_embed_fea_array, f)
