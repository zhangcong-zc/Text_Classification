# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time : 2020/4/5 16:00 
# @Author : Zhang Cong

import re
import os
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def data_generate(root_path, output_file_path):
    '''
    遍历文件夹，生成训练数据（原数据共18个类别，部分类别样本量过少，因此选出10个类别，每个类别选出5000个样本）
    :param root_path: 原始数据路径
    :param output_file_path: 筛选并进行格式转换后的路径
    :return:
    '''
    logging.info('Start Generate Data ...')
    # 类别
    label_num_dict = {'auto': 0, 'it': 0, 'health': 0, 'sports': 0, 'travel': 0,
                      'learning': 0, 'house': 0, 'yule': 0, 'women': 0, 'business': 0}

    output_file = open(output_file_path, mode='w', encoding='UTF-8')
    # 遍历文件夹中的全部文件
    for file_name in tqdm(os.listdir(root_path)):
        file_path = os.path.join(root_path, file_name)
        if file_path.endswith('.txt') or file_path.endswith('.TXT'):
            text = open(file_path, mode='r', encoding='GB18030').read()
            # 正则匹配所有的doc字段
            document_list = re.compile('<doc>.*?</doc>', re.DOTALL).findall(text)
            for document in document_list:
                # 从url字段抽取label信息
                url = re.compile('<url>.*?</url>', re.DOTALL).findall(document)[0].replace('<url>http://', '').replace('</url>', '')
                dot_index = str(url).index('.')
                label = url[: dot_index]
                # 抽取新闻title信息
                content_title = re.compile('<contenttitle>.*?</contenttitle>', re.DOTALL).findall(document)[0].replace('<contenttitle>', '').replace('</contenttitle>', '')
                # 抽取新闻content信息
                content = re.compile('<content>.*?</content>', re.DOTALL).findall(document)[0].replace('<content>', '').replace('</content>', '')
                # 过滤长度较短的文本
                if label in label_num_dict.keys() and len(content) > 20:
                    if label_num_dict[label] < 5000:    # 每个样本数量为5000
                        label_num_dict[label] += 1
                        output_file.write(label + '\t' + content_title + ' ' + content + '\n')

    output_file.close()
    print(label_num_dict)
    logging.info('Generate Data Success ...')


if __name__ == "__main__":
    root_path = 'F:/Data/SogouCS 搜狗新闻分类/SogouCS.reduced'
    output_file_path = './data/data.txt'
    data_generate(root_path, output_file_path)