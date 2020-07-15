# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong

from config import Config
import tensorflow as tf
from tensorflow.contrib import rnn

class Model():

    def __init__(self):
        self.config = Config()                                                                                  # 配置参数
        self.input_x = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-x')     # 输入文本
        self.input_y = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name='input-y')    # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                         # keep-prob

        # Embedding layer
        embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)            # dim:(batch_size, 100, 300)

        # Bi-LSTM/Bi-GRU
        cell_fw = self.get_rnn(self.config.rnn_type)        # 前向cell
        cell_bw = self.get_rnn(self.config.rnn_type)        # 后向cell
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedding_x, dtype=tf.float32)
        outputs = tf.concat(values=outputs, axis=2)         # 将前向cell和后向cell的结果进行concat拼接   dim:(batch_size, 100, 2*hidden_dim)

        # Attention  此处实现的是weight attention机制
        attention_w = tf.get_variable(shape=[2*self.config.hidden_dim, self.config.attention_size], dtype=tf.float32, name='attention-w')
        attention_b = tf.get_variable(initializer=0.1, dtype=tf.float32, name='attention-b')
        attention_uw = tf.get_variable(shape=[self.config.attention_size, 1], dtype=tf.float32, name='attention-uw')
        outputs = tf.transpose(outputs, [1, 0, 2])      # dim:(100, batch_size, 2*hidden_dim)

        attention_list = []
        for index in range(self.config.seq_length):     # 在每个时序上进行attention
            att = tf.tanh(tf.matmul(outputs[index], attention_w) + attention_b)     # weight attention  [batch_size, 128]
            att = tf.matmul(att, attention_uw)          # 全连接层   [batch_size, 1]
            attention_list.append(att)
        # attention_res = tf.concat(values=attention_list, axis=1)      # 与下面语句同样效果
        attention_res = tf.transpose(tf.squeeze(input=attention_list, axis=2), [1, 0])      # [batch_size, 100]
        # 归一化后的 attention score
        attention_score = tf.nn.softmax(logits=attention_res)
        attention_score = tf.expand_dims(input=attention_score, axis=-1)

        # 进行attention加权计算 [batch_size, 100, 256]  [batch_size, 100, 1]
        self.final_output = tf.transpose(outputs, [1, 0, 2]) * attention_score
        self.final_output = tf.reduce_sum(self.final_output, axis=1)        # [batch_size, 256]

        # 输出层
        self.logits = tf.layers.dense(inputs=self.final_output, units=self.config.num_classes, name='logits')
        self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')
        # 损失函数，交叉熵
        self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        self.loss = tf.reduce_mean(self.entropy, name='loss')
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


    def get_rnn(self, rnn_type):
        '''
        根据rnn_type创建RNN层
        :param rnn_type: RNN类型
        :return:
        '''
        if rnn_type == 'lstm':
            cell = rnn.LSTMCell(num_units=self.config.hidden_dim)
        else:
            cell = rnn.GRUCell(num_units=self.config.hidden_dim)
        cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=self.input_keep_prob)
        return cell



# Model()