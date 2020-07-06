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

        # embedding layer
        embedding = tf.get_variable(shape=[config.vocab_size, config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)        # dim:(batch_size, 100, 300)

        # 遍历多个大小的卷积核
        pooling_result = []
        for kernel_size in config.kernel_sizes:
            # CNN layer   # dim:(batch_size, seq_length-kernel_size+1, hidden_dim)
            conv = tf.layers.conv1d(inputs=embedding_x, filters=config.hidden_dim, kernel_size=kernel_size, name='conv-'+str(kernel_size))
            # global max pooling    # dim:(batch_size, 1, hidden_dim)
            max_pooling = tf.layers.max_pooling1d(inputs=conv, pool_size=config.seq_length-kernel_size+1, strides=1)
            max_pooling = tf.squeeze(input=max_pooling, axis=1)         # dim:(batch_size, hidden_dim)
            pooling_result.append(max_pooling)
        # 将多个卷积核大小的结果进行拼接
        max_pooling_all = tf.concat(values=pooling_result, axis=1)

        # 全连接层，后接dropout及relu
        fc = tf.layers.dense(inputs=max_pooling_all, units=128, name='fc1')
        fc = tf.layers.dropout(inputs=fc, rate=self.input_keep_prob)
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