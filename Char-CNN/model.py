# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong

from config import Config
import tensorflow as tf

class Model():

    def __init__(self):
        self.config = Config()                                                                                  # 配置参数
        self.input_x = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-x')     # 输入文本
        self.input_y = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name='input-y')    # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                         # keep-prob

        # Embedding 层
        embedding = tf.get_variable(shape=[self.config.char_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')
        x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)              # dim:(batch_size, 100, 300)

        # 多个卷积层（原论文中为6层conv + 3层 pooling）
        for layer in self.config.conv_layers:
            # 一维卷积
            conv = tf.layers.conv1d(inputs=x, filters=layer[0], kernel_size=layer[1], padding='valid', use_bias=True)
            # activation
            x = tf.nn.relu(conv)
            if layer[2] != None:    # 判断是否进行pooling操作
                x = tf.layers.max_pooling1d(inputs=x, pool_size=layer[2], strides=layer[2])

        # 维度变换  # dim:(batch_size, fc_dim)
        x = tf.reshape(tensor=x, shape=[-1, x.get_shape()[1]*x.get_shape()[2]])

        # 全连接层 后接dropout
        for layer in self.config.fully_layers:
            fc = tf.layers.dense(inputs=x, units=layer, use_bias=True)
            x = tf.nn.dropout(x=fc, keep_prob=self.config.keep_prob)

        # 输出层
        self.logits = tf.layers.dense(inputs=x, units=self.config.num_classes, name='logits')
        self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


# Model()