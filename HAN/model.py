# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong

from config import Config
import tensorflow as tf
from tensorflow.contrib import rnn

class Model():

    def __init__(self):
        self.config = Config()      # 配置参数
        # 输入文本 [batch_size, 文本中句子数量, 句子中词汇数量]
        self.input_x = tf.placeholder(shape=[None, self.config.document_sentence_num, self.config.sentence_word_num], dtype=tf.int32, name='input-x')
        self.input_y = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.float32, name='input-y')      # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')         # keep-prob

        # Embedding layer
        embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)     # [batch_size, document_sentence_num, sentence_word_num, embedding_dim]

        # word-level处理
        with tf.name_scope(name='word-level') as scope:
            # Word level bidirection rnn
            word_level_rnn_input = tf.reshape(tensor=embedding_x, shape=[-1, self.config.sentence_word_num, self.config.embedding_dim])
            word_level_rnn_output = self.bidirection_rnn(input_data=word_level_rnn_input, rnn_type=self.config.rnn_type, scope=scope)
            # word level attention
            word_level_attention_output = self.attention(word_level_rnn_output, self.config.attention_size)
            word_level_attention_output = tf.nn.dropout(word_level_attention_output, keep_prob=self.input_keep_prob)

        # sentence-level处理
        with tf.name_scope(name='sentence-level') as scope:
            # sentence level bidirection rnn
            sentence_level_rnn_input = tf.reshape(tensor=word_level_attention_output, shape=[-1, self.config.document_sentence_num, 2*self.config.hidden_dim])
            sentence_level_rnn_output = self.bidirection_rnn(input_data=sentence_level_rnn_input, rnn_type=self.config.rnn_type, scope=scope)
            # sentence level attention
            sentence_level_attention_output = self.attention(sentence_level_rnn_output, self.config.attention_size)
            sentence_level_attention_output = tf.nn.dropout(sentence_level_attention_output, keep_prob=self.input_keep_prob)

        # fully connect layer
        self.logits = tf.layers.dense(inputs=sentence_level_attention_output, units=self.config.num_classes, name='logits')
        # 输出层
        self.predict = tf.argmax(tf.nn.softmax(self.logits), axis=1, name='predict')
        # 损失函数
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        self.loss = tf.reduce_mean(self.cross_entropy, name='loss')
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))


    def attention(self, input_data, attention_size):
        '''
        self-attention层，此处为weight attention机制
        :param input_data: 输入的向量形式数据
        :param attention_size: attention隐藏层大小
        :return:
        '''
        # 全连接层，调整隐藏层大小
        attention_res = tf.layers.dense(inputs=input_data, units=attention_size, activation=tf.nn.tanh, use_bias=True)
        attention_res = tf.layers.dense(inputs=attention_res, units=1, use_bias=False)
        # 降维，删除最后一个维度
        attention_res = tf.squeeze(attention_res, -1)
        # 对attention score进行归一化
        attention_score = tf.nn.softmax(attention_res, 1)
        # 进行attention weight 相乘操作
        attention_res = tf.reduce_sum(input_data * tf.expand_dims(input=attention_score, axis=-1), axis=1)
        return attention_res


    def bidirection_rnn(self, input_data, rnn_type, scope):
        '''
        双向RNN层，可根据rnn_type指定类型
        :param input_data: 输入数据
        :param rnn_type: rnn类型，LSTM/GRU
        :return:
        '''
        cell_fw = self.get_rnn(rnn_type)
        cell_bw = self.get_rnn(rnn_type)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=input_data, dtype=tf.float32, scope=scope)
        outputs = tf.concat(values=outputs, axis=-1)
        return outputs


    def get_rnn(self, rnn_type):
        '''
        根据rnn_type获取RNN Cell
        :param rnn_type: rnn类型，LSTM/GRU
        :return:
        '''
        if rnn_type=='lstm':
            cell = rnn.LSTMCell(num_units=self.config.hidden_dim)
        else:
            cell = rnn.GRUCell(num_units=self.config.hidden_dim)
        cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=self.input_keep_prob)
        return cell



# Model()