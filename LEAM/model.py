# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:30
# @Author: Zhang Cong

from config import Config
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class Model():

    def __init__(self):
        self.config = Config()                                                                                  # 配置参数
        self.input_x = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-x')     # 输入文本
        self.input_y = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name='input-y')    # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                         # keep-prob

        # content embedding layer 将文本进行向量化表示
        embedding_data = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding-data')
        embedding_x = tf.nn.embedding_lookup(params=embedding_data, ids=self.input_x)

        # label embedding layer 将标签进行向量化表示
        embedding_label = tf.get_variable(shape=[self.config.num_classes, self.config.embedding_dim], dtype=tf.float32, name='embedding-label')

        # 对输入文本使用bidirection_rnn进行编码
        rnn_output = self.bidirection_rnn(input_data=embedding_x, rnn_type=self.config.rnn_type)

        # content和label做attention
        output = self.context_label_attention(input_data=rnn_output, embedding_label=embedding_label, attention_size=self.config.attention_size)

        # content 输出层
        self.logits_data = tf.layers.dense(inputs=output, units=self.config.num_classes, name='logits-data')
        self.predict = tf.argmax(tf.nn.softmax(logits=self.logits_data), axis=1, name='predict-data')
        # content 损失函数
        self.cross_entropy_data = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits_data)
        self.loss_data = tf.reduce_mean(self.cross_entropy_data, name='loss-data')

        # 创建一个大小为num_classes * num_classes的矩阵，对角线数值都为1
        class_y = tf.constant(shape=[self.config.num_classes, self.config.num_classes], value=np.identity(self.config.num_classes), dtype=tf.float32, name='class-y')
        # label 输出层
        self.logit_label = tf.layers.dense(inputs=embedding_label, units=self.config.num_classes)
        # label 损失函数
        self.cross_entropy_label = tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=self.logit_label)
        self.loss_label = tf.reduce_mean(self.cross_entropy_label)
        # 全局损失函数
        self.loss = 0.5*self.loss_data + 0.5*self.loss_label
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, axis=1), self.predict_data)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))



    def context_label_attention(self, input_data, embedding_label, attention_size):
        '''
        计算context和label的attention值，此处为点乘attention机制
        :param input_data: 输入的embedding后的文本数据
        :param embedding_label: 输入的embedding后的标签数据
        :param attention_size: attention隐藏层大小
        :return:
        '''
        # 全连接层，将input_data缩放到attention_size大小 [batch_size, seq_length, attention_size]
        input_data = tf.layers.dense(inputs=input_data, units=attention_size)
        # 全连接层，embedding_label [num_classes, attention_size]
        embedding_label = tf.layers.dense(inputs=embedding_label, units=attention_size)
        embedding_label = tf.transpose(embedding_label, [1, 0])    # [attention_size, num_classes]

        # L2 正则化，均值为0，方差为1
        input_data = tf.nn.l2_normalize(input_data, axis=-1)
        embedding_label = tf.nn.l2_normalize(embedding_label, axis=0)

        # G = tf.einsum('ijk,kl->ijl', input_data, embedding_label)     G为attention向量
        G = tf.matmul(tf.reshape(tensor=input_data, shape=[-1, input_data.get_shape()[-1]]), embedding_label)
        G = tf.reshape(tensor=G, shape=[-1, self.config.seq_length, self.config.num_classes]) # [batch_size, seq_length, num_classes]
        G = tf.expand_dims(input=G, axis=-1)    # [batch_size, seq_length, num_classes, 1]

        # conv层
        G = tf.layers.conv2d(inputs=G, filters=1, kernel_size=[8, 1], strides=[1, 1], padding='same', name='conv-w')
        G = tf.nn.relu(G)
        G = tf.squeeze(input=G, axis=-1)        # [batch_size, seq_length, num_classes]
        # max pooling 最大池化
        max_G = tf.reduce_max(input_tensor=G, axis=-1, keep_dims=True)
        # 对attention值进行归一化
        soft_max_G = tf.nn.softmax(max_G, 1)
        # 进行attention计算
        output = tf.reduce_sum(input_tensor=input_data*soft_max_G, axis=1)
        return output


    def bidirection_rnn(self, input_data, rnn_type):
        '''
        双向RNN层，可根据rnn_type指定类型
        :param input_data: 输入数据
        :param rnn_type: rnn类型，LSTM/GRU
        :return:
        '''
        cell_fw = self.get_rnn(rnn_type)
        cell_bw = self.get_rnn(rnn_type)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=input_data, dtype=tf.float32)
        output = tf.concat(values=outputs, axis=-1)
        return output


    def get_rnn(self, rnn_type):
        '''
        根据rnn_type获取RNN Cell
        :param rnn_type: rnn类型，LSTM/GRU
        :return:
        '''
        if rnn_type == 'lstm':
            cell = rnn.LSTMCell(num_units=self.config.hidden_dim)
        else:
            cell = rnn.GRUCell(num_units=self.config.hidden_dim)
        cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.config.keep_prob)
        return cell


# Model()