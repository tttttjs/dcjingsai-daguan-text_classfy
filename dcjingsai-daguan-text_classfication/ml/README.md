# dcjingsai_text_classification
[“达观杯”文本智能处理挑战赛](http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E5%9C%88.html)

## 传统模型部分
### 代码说明
- Config目录
	- param_config.py: 全局配置参数
- Input目录
	- process_data.py: 产出skf，train_labels，test_ids等中间数据
- Feat目录
	- gen_word_basic_tfidf_fea.py: 产出lgb训练所用的word级别tfidf特征
	- gen_char_basic_tfidf_fea.py: 产出lgb训练所用的char级别tfidf特征
	- gen_word_tfidf_fea3.py: 产出svc训练所用的word级别tfidf特征
	- gen_char_tfidf_fea3.py: 产出svc训练所用的char级别tfidf特征
	- gen_average_embed_fea.py: 产出词级别的均值embedding特征
- LDA目录
	- gen_tf.py: 产出训练LDA所需的tf数据
	- lda_train_sklearn_tf.py: 产出LDA特征
- Word2vec目录
	- gensim_word2vec.py: 训练word2vec，产出embedding向量
- SVM_LinearSVC目录
	- svc_train_word.py: 训练词级别svc模型
	- svc_train_char.py: 训练字符级别svc模型
- LightGBM目录
	- lgb_train_word_lda_embed.py: 训练词级别lgb模型
	- lgb_tran_char.py: 训练线下字符级别lgb模型
	- lgb_train_on_char.py: 训练线上字符级别lgb模型
- Ensemble目录
	- ensemble_lgb_svc2.py: 线下确定融合系数，产出最终融合结果