# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:21:40 2018

@author: haota
"""

# 载入数据集
# 首先我们使用已经训练好的词向量模型，词向量模型如何训练
# 首先引入2个数据集合，400 000的词典以及400 000 x 50维的嵌入矩阵
import numpy as np
words_list = np.load('wordsList.npy')
print('载入word列表')
words_list = words_list.tolist()  # 转化为list
words_list = [word.decode('UTF-8') for word in words_list]
word_vectors = np.load('wordVectors.npy')
print('载入文本向量')

# 检查数据
print(len(words_list))
print(word_vectors.shape)

# 构造整个训练集索引
# 需要先可视化和分析数据的情况从而确定并设置最好的序列长度
import os
from os.path import isfile,join
pos_files = ['pos/' + f for f in os.listdir('pos/') if isfile(join('pos/',f))] #正例
neg_files = ['neg/' + f for f in os.listdir('neg/') if isfile(join('neg/',f))] #负例
num_words = []
for pf in pos_files:
    with open(pf,"r",encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完毕')
for nf in neg_files:
    with open(nf,"r",encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完毕')
num_files = len(num_words)
print('文件总数',num_files)
print('所有词的数量',sum(num_words))
print('平均文件词的长度',sum(num_words)/len(num_words))
#进行可视化
import matplotlib
#matplotlib.use('qt4agg')

#指定默认字体,解决matplot显示中文的问题
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.pyplot.hist(num_words,50,facecolor='g')
matplotlib.pyplot.xlabel('文本长度')
matplotlib.pyplot.ylabel('频次')
matplotlib.pyplot.show()
# 可见大部分文本长度在230以内，保守起见，设置序列最大长度为300
#max_seq_len = 300

# 将文本生成一个索引矩阵，并且得到25000 x 300 的矩阵
import re
strip_special_chars = re.compile('[^A-Za-z0-9 ]+')
def cleanSentences(string):
    string = string.lower().replace("<br />"," ") #字符替换
    return re.sub(strip_special_chars,"",string.lower())

max_seq_num = 250
ids = np.zeros((num_files,max_seq_num),dtype='int32')
file_count = 0 # 文件数目
for pf in pos_files:
    with open(pf,"r",encoding='utf-8') as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[file_count][indexCounter] = words_list.index(word) #该单词在词汇表中的索引值放入矩阵中
            except ValueError:
                ids[file_count][indexCounter] = 399999 #未知的词
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
    file_count += 1

for nf in neg_files:
    with open(nf,"r",encoding='utf-8') as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[file_count][indexCounter] = words_list.index(word) #该单词在词汇表中的索引值放入矩阵中
            except ValueError:
                ids[file_count][indexCounter] = 399999 #未知的词
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
    file_count += 1

# 保存到文件
np.save('idsMatrix',ids)  #保存为npy文件
