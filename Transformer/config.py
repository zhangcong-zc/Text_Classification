# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong


# 模型配置参数
class Config():
    def __init__(self):
        self.original_data_path = './data/data.txt'
        self.stopwords_path = './data/stopwords.txt'
        self.preprocess_path = './data/preprocess_data.txt'
        self.vocab_path = './data/vocab.txt'
        self.label_path = './data/label.txt'
        self.model_save_path = './save_model/'
        self.seq_length = 100
        self.num_classes = 10
        self.batch_size = 32
        self.keep_prob = 0.5
        self.epochs = 10
        self.vocab_size = 5000
        self.hidden_dim = 128
        self.embedding_dim = 128
        self.num_blocks = 6
        self.num_heads = 8
        self.learning_rate = 1e-3
        self.train_test_split_value = 0.9

