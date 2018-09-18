#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
@file:	 gensim_word2vec.py
@author: liaolin (liaolin@baidu.com)
@date:	 2018年09月09日 星期日 12时36分04秒
@brief:	 
"""
import pandas as pd

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

embed_size = 300

df_train = pd.read_csv('../Input/train_set.csv', sep=',')
df_test = pd.read_csv('../Input/test_set.csv', sep=',')
df = pd.concat([df_train, df_test], ignore_index=True)
df['word_list'] = df['word_seg'].apply(lambda x: [w for w in x.split(' ')])

all_word_list = df['word_list'].values

model = Word2Vec(all_word_list, size=embed_size, window=5, min_count=1, workers=32)

word_vectors = model.wv

word_vectors_file = '../Feat/word_vectors_size{}'.format(embed_size)
word_vectors.save(word_vectors_file)
