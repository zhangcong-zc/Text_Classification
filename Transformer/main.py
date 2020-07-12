# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/4/5 17:31
# @Author: Zhang Cong

import os
import jieba
import logging
import numpy as np
from tqdm import tqdm
from model import Model
import tensorflow as tf
from config import Config
from collections import Counter
import sklearn.metrics as metrics

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

config = Config()
# GPU配置信息
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                        # 设置当前使用的GPU设备仅为0号设备
gpuConfig = tf.ConfigProto()
gpuConfig.allow_soft_placement = True                           #设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
gpuConfig.gpu_options.allow_growth = True                       #设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.8     #程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值

# 模型训练
class Train():
    def __init__(self):
        # 实例化模型结构
        self.model = Model()
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        # 数据集预处理
        if not os.path.exists(config.preprocess_path):
            pre_process(config.original_data_path, config.preprocess_path)
        sentences, labels = load_dataset(config.preprocess_path)  # 加载数据集
        # 构建词汇映射表
        if not os.path.exists(config.vocab_path):
            build_vocab(sentences, config.vocab_path)
        word_to_id = read_vocab(config.vocab_path)  # 读取词汇表及其映射关系
        # 构建类别映射表
        if not os.path.exists(config.label_path):
            build_label(labels, config.label_path)
        label_to_id = read_label(config.label_path)  # 读取类别表及其映射关系

        # 构建训练数据集
        data_sentences, data_labels = data_transform(sentences, labels, word_to_id, label_to_id)
        # 训练集、测试集划分
        split_index = int(len(data_sentences) * config.train_test_split_value)
        train_data, test_data = data_sentences[: split_index], data_sentences[split_index:]
        train_label, test_label = data_labels[: split_index], data_labels[split_index:]

        # 打印训练、测试数据量，数据与标签量是否相等
        logging.info('Train Data: {}'.format(np.array(train_data).shape))
        logging.info('Train Label: {}'.format(np.array(train_label).shape))
        logging.info('Test Data: {}'.format(np.array(test_data).shape))
        logging.info('Test Label: {}'.format(np.array(test_label).shape))

        # 配置Saver
        saver = tf.train.Saver()
        if not os.path.exists(config.model_save_path):  # 如不存在相应文件夹，则创建
            os.mkdir(config.model_save_path)

        # 模型训练
        best_f1_score = 0  # 初始best模型的F1值
        for epoch in range(1, config.epochs + 1):
            train_accuracy_list = []  # 存储每个epoch的accuracy
            train_loss_list = []  # 存储每个epoch的loss
            # 将训练数据进行 batch_size 切分
            batch_train_data, batch_train_label = creat_batch_data(train_data, train_label, config.batch_size)
            for step, (batch_x, batch_y) in tqdm(enumerate(zip(batch_train_data, batch_train_label))):
                feed_dict = {self.model.input_x: batch_x,
                             self.model.input_y: batch_y,
                             self.model.input_keep_prob: config.keep_prob}
                train_accuracy, train_loss, _ = self.sess.run(
                    [self.model.accuracy, self.model.loss, self.model.optimizer], feed_dict=feed_dict)
                train_accuracy_list.append(train_accuracy)
                train_loss_list.append(train_loss)
                # 完成一个epoch的训练，输出训练数据的mean accuracy、mean loss
            logging.info('Train Epoch: %d , Loss: %.6f , Acc: %.6f' % (epoch,
                                                                       float(np.mean(np.array(train_loss_list))),
                                                                       float(
                                                                           np.mean(np.array(train_accuracy_list)))))
            # 模型验证
            test_accuracy_list = []  # 存储每个epoch的accuracy
            test_loss_list = []  # 存储每个epoch的loss
            test_label_list = []  # 存储数据的true label
            test_predictions = []  # 存储模型预测出的label
            batch_test_data, batch_test_label = creat_batch_data(test_data, test_label, config.batch_size)
            for (batch_x, batch_y) in tqdm(zip(batch_test_data, batch_test_label)):
                feed_dict = {self.model.input_x: batch_x,
                             self.model.input_y: batch_y,
                             self.model.input_keep_prob: 1.0}
                test_predict, test_accuracy, test_loss = self.sess.run(
                    [self.model.predict, self.model.accuracy, self.model.loss], feed_dict=feed_dict)
                test_accuracy_list.append(test_accuracy)
                test_loss_list.append(test_loss)
                test_label_list.extend(batch_y)
                test_predictions.extend(test_predict)

            # 获取最大score所在的index
            true_y = [np.argmax(label) for label in test_label_list]
            # 计算模型F1 score
            f1_score = metrics.f1_score(y_true=np.array(true_y), y_pred=np.array(test_predictions),
                                        average='weighted')
            # 详细指标报告  Precision， Recall， F1
            report = metrics.classification_report(y_true=np.array(true_y), y_pred=np.array(test_predictions))

            logging.info('Test Epoch: %d , Loss: %.6f , Acc: %.6f , F1 Socre: %.6f' % (epoch,
                                                                                       float(np.mean(np.array(
                                                                                           test_loss_list))),
                                                                                       float(np.mean(np.array(
                                                                                           test_accuracy_list))),
                                                                                       f1_score))
            print('Report: \n', report)
            # 当前epoch产生的模型F1值超过最好指标时，保存当前模型
            if best_f1_score < f1_score:
                best_f1_score = f1_score
                saver.save(sess=self.sess, save_path=config.model_save_path)
                logging.info('Save Model Success ...')


# 模型预测
class Predict():
    def __init__(self):
        # 实例化并加载模型
        self.model = Model()
        self.sess = tf.Session(config=gpuConfig)
        self.saver = tf.train.Saver()
        self.saver.restore(sess=self.sess, save_path=config.model_save_path)

    # 结果预测
    def predict(self, sentence):
        # 加载词汇->ID映射表
        words, word_to_id = read_vocab(config.vocab_path)
        stopwords = [word.replace('\n', '').strip() for word in open(config.stopwords_path, encoding='UTF-8')]
        # 分词，去除停用词
        sentence_seg = [word for word in list(jieba.cut(sentence)) if word not in stopwords and not word.isdigit()]
        # 将词汇映射为ID
        sentence_id = [word_to_id[word] for word in sentence_seg if word in word_to_id]
        # 对句子进行padding
        sentence_length = len(sentence_id)
        if sentence_length > config.seq_length:
            sentence_id = sentence_id[: config.seq_length]
        else:
            sentence_id.extend([0] * (config.seq_length - sentence_length))

        feed_dict = {self.model.input_x: [sentence_id], self.model.input_keep_prob: 1.0}
        predict = self.sess.run(self.model.predict, feed_dict=feed_dict)[0]

        return predict


def text_processing(text):
    '''
    文本数据预处理，分词，去除停用词
    :param text: 文本数据sentence
    :return: 以空格为分隔符进行分词/分字结果
    '''
    # 删除（）里的内容
    # text = re.sub('（[^（.]*）', '', text)
    # 只保留中文部分
    text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
    # 利用jieba进行分词
    words = list(jieba.cut(text))
    # 不分词
    # words = [x for x in ''.join(text)]
    return ' '.join(words)


def pre_process(data_path, preprocess_path):
    '''
        原始数据预处理
        :param data_path: 原始文本文件路径
        :param preprocess_path: 预处理后的数据存储路径
        :return:
        '''
    # 加载停用词表
    logging.info('Start Preprocess ...')
    preprocess_file = open(preprocess_path, mode='w', encoding='UTF-8')
    # 加载停用词表
    stopwords = [word.replace('\n', '').strip() for word in open(config.stopwords_path, encoding='UTF-8')]
    for line in tqdm(open(data_path, encoding='UTF-8')):
        label_sentence = str(line).strip().replace('\n', '').split('\t')    # 去除收尾空格、结尾换行符\n、使用\t切分
        label = label_sentence[0]
        sentence = label_sentence[1].replace('\t', '').replace('\n', '').replace(' ', '')   # 符号过滤
        sentence = [word for word in text_processing(sentence).split(' ') if word not in stopwords and not word.isdigit()]
        preprocess_file.write(label + '\t' + ' '.join(sentence) + '\n')

    preprocess_file.close()


def load_dataset(data_path):
    '''
    从本地磁盘加载经过预处理的数据集，避免每次都进行预处理操作
    :param data_path: 预处理好的数据集路径
    :return: 句子列表，标签列表
    '''
    sentences = []
    labels = []
    # 加载停用词表
    logging.info('Load Dataset ...')
    for line in tqdm(open(data_path, encoding='UTF-8')):
        try:
            label_sentence = str(line).strip().replace('\n', '').split('\t')
            label = label_sentence[0]                           # 标签
            sentence = label_sentence[1]                        # 以空格为切分的content
            sentence = [word for word in sentence.split(' ')]
            sentences.append(sentence)
            labels.append(label)
        except:
            logging.info('Load Data Error ... msg: {}'.format(line))    # 部分数据去除英文和数字后为空，跳过异常
            continue

    return sentences, labels


def build_vocab(input_data, vocab_path):
    '''
    根据数据集构建词汇表，存储到本地备用
    :param input_data: 全部句子集合 [n] n为数据条数
    :param vocab_path: 词表文件存储路径
    :return:
    '''
    logging.info('Build Vocab ...')
    all_data = []       # 全部句子集合
    for content in input_data:
        all_data.extend(content)

    counter = Counter(all_data)     # 词频统计
    count_pairs = counter.most_common(config.vocab_size - 2)    # 对词汇按次数进行降序排序
    words, _ = list(zip(*count_pairs))              # 将(word, count)元祖形式解压，转换为列表list
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<UNK>'] + list(words)  # 增加一个OOV标识的编码
    words = ['<PAD>'] + list(words)  # 增加一个PAD标识的编码
    open(vocab_path, mode='w', encoding='UTF-8').write('\n'.join(words) + '\n')


def build_label(input_label, label_path):
    '''
    根据标签集构建标签表，存储到本地备用
    :param input_label: 全部标签集合
    :param label_path: 标签文件存储路径
    :return:
    '''
    logging.info('Build Label ...')
    all_label = set(input_label)
    open(label_path, mode='w', encoding='UTF-8').write('\n'.join(all_label))


def read_vocab(vocab_path):
    """
    读取词汇表，构建 词汇-->ID 映射字典
    :param vocab_path: 词表文件路径
    :return: 词表，word_to_id
    """
    words = [word.replace('\n', '').strip() for word in open(vocab_path, encoding='UTF-8')]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def read_label(label_path):
    '''
    读取类别表，构建 类别-->ID 映射字典
    :param label_path: 类别文件路径
    :return: 类别表，label_to_id
    '''
    labels = [label.replace('\n', '').strip() for label in open(label_path, encoding='UTF-8')]
    label_to_id = dict(zip(labels, range(len(labels))))

    return label_to_id


def data_transform(input_data, input_label, word_to_id, label_to_id):
    '''
    数据预处理，将文本和标签映射为ID形式
    :param input_data: 文本数据集合
    :param input_label: 标签集合
    :param word_to_id: 词汇——ID映射表
    :param label_to_id: 标签——ID映射表
    :return: ID形式的文本，ID形式的标签
    '''
    logging.info('Sentence Trans To ID ...')
    sentence_id = []
    # 将文本转换为ID表示[1,6,2,3,5,8,9,4]
    for sentence in tqdm(input_data):
        sentence_temp = []
        for word in sentence:
            if word in word_to_id:
                sentence_temp.append(word_to_id[word])
            else:
                sentence_temp.append(word_to_id['<UNK>'])

        # 对文本长度进行padding填充
        sentence_length = len(sentence_temp)
        if sentence_length > config.seq_length:
            sentence_temp = sentence_temp[: config.seq_length]
        else:
            sentence_temp.extend([word_to_id['<PAD>']] * (config.seq_length - sentence_length))
        sentence_id.append(sentence_temp)


    # 将标签转换为ID形式
    logging.info('Label Trans To One-Hot ...')
    label_id = []
    for label in tqdm(input_label):
        label_id_temp = np.zeros([config.num_classes])
        if label in label_to_id:
            label_id_temp[label_to_id[label]] = 1
            label_id.append(label_id_temp)

    # shuffle
    indices = np.random.permutation(np.arange(len(sentence_id)))
    datas = np.array(sentence_id)[indices]
    labels = np.array(label_id)[indices]

    return datas, labels


def creat_batch_data(input_data, input_label, batch_size):
    '''
    将数据集以batch_size大小进行切分
    :param input_data: 数据列表
    :param input_label: 标签列表
    :param batch_size: 批大小
    :return:
    '''
    max_length = len(input_data)            # 数据量
    max_index = max_length // batch_size    # 最大批次
    # shuffle
    indices = np.random.permutation(np.arange(max_length))
    data_shuffle = np.array(input_data)[indices]
    label_shuffle = np.array(input_label)[indices]

    batch_data, batch_label = [], []
    for index in range(max_index):
        start = index * batch_size                              # 起始索引
        end = min((index + 1) * batch_size, max_length)         # 结束索引，可能为start + batch_size 或max_length
        batch_data.append(data_shuffle[start: end])
        batch_label.append(label_shuffle[start: end])

        if (index + 1) * batch_size > max_length:               # 如果结束索引超过了数据量，则结束
            break

    return batch_data, batch_label


if __name__ == "__main__":
    # 训练
    Train().train()

    # # 预测
    # predictor = Predict()
    # while True:
    #     sentence = input('输入：')
    #     result = predictor.predict(sentence)
    #     print(result)