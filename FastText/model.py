# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong

import tensorflow as tf
from config import Config

class Model():

    def __init__(self):
        config = Config()                                                                                   # 配置参数
        self.input_x = tf.placeholder(shape=[None, config.seq_length], dtype=tf.int32, name='input-x')      # 输入文本
        self.input_y = tf.placeholder(shape=[None, config.num_classes], dtype=tf.int32, name='input-y')     # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                     # keep-prob

        # Embedding layer
        embedding = tf.get_variable(shape=[config.vocab_size, config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)        # dim:(batch_size, 100, 300)

        # dropout层   对表征后的词向量进行dropout操作
        dropout_x = tf.layers.dropout(inputs=embedding_x, rate=self.input_keep_prob)
        # 对文本中的词向量进行求和平均
        embedding_mean_x = tf.reduce_mean(input_tensor=dropout_x, axis=1)     # dim:(batch_size, 300)

        # 全连接层，后接dropout及relu
        fc = tf.layers.dense(inputs=embedding_mean_x, units=128, name='fc1')
        fc = tf.nn.relu(fc)
        # 输出层
        logits = tf.layers.dense(inputs=fc, units=config.num_classes, name='logits')
        self.predict = tf.argmax(input=tf.nn.softmax(logits), axis=1, name='predict')        # 预测类别
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss=self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


# Model()