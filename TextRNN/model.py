# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong

from config import Config
import tensorflow as tf
import tensorflow.contrib as contrib

class Model():

    def __init__(self):
        self.config = Config()                                                                                  # 配置参数
        self.input_x = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-x')     # 输入文本
        self.input_y = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name='input-y')    # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                         # keep-prob

        # embedding layer
        embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)            # dim:(batch_size, 100, 300)
        # 创建多个叠加的RNN层，并将cell组成一个list
        cells = [self.get_rnn(self.config.rnn_type) for _ in range(self.config.num_layers)]
        rnn_cell = contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_x, dtype=tf.float32)
        # 取RNN最后一个时序的输出结果作为最终结果
        last = output[ :, -1, :]

        # 全连接层，后接dropout及relu
        fc = tf.layers.dense(inputs=last, units=self.config.hidden_dim, name='fc1')
        fc = tf.nn.dropout(fc, keep_prob=self.config.keep_prob)
        fc = tf.nn.relu(fc)

        # 输出层
        self.logits = tf.layers.dense(inputs=fc, units=self.config.num_classes, name='logits')
        self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')     # 预测类别
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


    # 根据rnn_type创建RNN层
    def get_rnn(self, rnn_type):
        if rnn_type == 'lstm':
            cell = contrib.rnn.BasicLSTMCell(num_units=self.config.hidden_dim, state_is_tuple=True)
        else :
            cell = contrib.rnn.GRUCell(num_units=self.config.hidden_dim)
        return contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=self.input_keep_prob)



# Model()