# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/21 21:30
# @Author: Zhang Cong

import numpy as np
import tensorflow as tf
from config import Config

class Model():

    def __init__(self):
        self.config = Config()                                                                                  # 配置参数
        self.input_x = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-x')     # 输入文本
        self.input_y = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.int32, name='input-y')    # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                         # keep-prob

        # Embedding layer
        embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)            # dim:(batch_size, 100, 128)

        # 位置编码 Positional Encoding
        N = tf.shape(embedding_x)[0]       # batch_size
        T = self.config.seq_length         # seq_length
        position_embedding =  self.positional_encoding(N, T, num_units=self.config.embedding_dim, zero_pad=False, scale=False, scope="position_encode")
        embedding_x += position_embedding   # 词汇的embedding + position embedding

        # Dropout
        self.enc = tf.layers.dropout(embedding_x, rate=self.config.keep_prob)

        # Blocks (原论文中 6层multi-head_attention, 每层8个head)
        for i in range(self.config.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                self.enc = self.multihead_attention(queries=self.enc,
                                                    keys=self.enc,
                                                    num_units=self.config.hidden_dim,
                                                    num_heads=self.config.num_heads,
                                                    dropout_rate=self.config.keep_prob,
                                                    causality=False)
                # Feed Forward
                self.enc = self.feedforward(self.enc, num_units=[4 * self.config.hidden_dim, self.config.hidden_dim])

        # 将特征进行拼接
        self.enc = tf.reshape(self.enc, [-1, self.config.seq_length * self.config.hidden_dim, 1])      # [batch_size, seq_length*hidden_dim, 1]
        self.enc = tf.squeeze(self.enc, -1)     # [batch_size, seq_length*hidden_dim]

        # 初始化 weight and bias
        fc_w = tf.Variable(tf.truncated_normal([self.config.seq_length * self.config.hidden_dim, self.config.num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([self.config.num_classes]), name='fc_b')

        # 定义L2正则损失
        l2_loss = tf.nn.l2_loss(fc_w) + tf.nn.l2_loss(fc_b)

        # 输出层
        self.logits = tf.matmul(self.enc, fc_w) + fc_b
        self.score = tf.nn.softmax(self.logits, name='score')
        self.predict = tf.argmax(self.score, 1, name="predict")
        # 损失函数，交叉熵
        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)

        l2_reg_lambda = 0.01    # L2正则损失权重，避免模型过于复杂，缓解过拟合
        # 整体损失函数 = 经验损失 + 正则损失
        self.loss = self.cost + l2_reg_lambda * l2_loss
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, axis=1), self.predict)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


    def positional_encoding(self, N, T, num_units, zero_pad=True, scale=True, scope="positional_encoding", reuse=None):
        '''Sinusoidal Positional_Encoding.  位置编码函数
        Args:
          inputs: A 2d Tensor with shape of (N, T).
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
        Returns:
            A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # position index
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])      # [N, seq_length]
            # First part of the PE function: sin and cos argument   第一步：先计算sin和cos的参数
            position_enc = np.array([
                [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
                for pos in range(T)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i         # 偶数位使用sin进行计算
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1       # 奇数位使用cos进行计算

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)

            if zero_pad:    # 是否将每行的第一个数替换为0
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

            if scale:       # 是否进行缩放 num_units 平方根
                outputs = outputs * num_units ** 0.5
            outputs = tf.cast(outputs, dtype=tf.float32)

            return outputs


    def multihead_attention(self, queries, keys, num_units=None, num_heads=8, dropout_rate=0, causality=False, scope="multihead_attention", reuse=None):
        '''Applies multihead attention.  多头注意力函数
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:       # 如果num_units未指定，则使用query的最后一个维度作为num_units
                num_units = queries.get_shape().as_list[-1]

            # Linear projections    线性映射 生成Q、K、V三个attention matrix
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat  将上步生成的attention matrix切分为num_heads个小矩阵，并在第一维度进行拼接
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication    Q和K进行矩阵相乘
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            # Scale     数值缩放 sqrt embedding dim
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)
            # Weighted sum      加权求和
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            # Restore shape     将多头的结果进行concat
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            # Residual connection   残差链接
            outputs += queries
            # Normalize     标准化
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs


    def feedforward(self, inputs, num_units=[2048, 512], scope="multihead_attention", reuse=None):
        '''Point-wise feed forward net.  前向网络层
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs
            # Normalize
            outputs = self.normalize(outputs)

        return outputs


    def normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        '''Applies layer normalization.    标准化使均值为0，方差为1
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs


# Model()