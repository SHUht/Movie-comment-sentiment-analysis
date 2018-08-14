# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:29:36 2018

@author: haota
"""
import numpy as np
#from dataset import word_vectors # 从数据集引入词向量
word_vectors = np.load('wordVectors.npy')
# 模型参数设置
batch_size = 24  
lstm_units = 64
num_labels = 2
iterations = 30001
num_dimensions = 50
max_seq_num = 250
learning_rate = 0.001
inference_mode = True
ids = np.load('idsMatrix.npy')
# 辅助函数，返回一个数据集的迭代器，用于返回一批训练集合
from random import randint
def get_train_batch():
    labels = []
    arr = np.zeros([batch_size,max_seq_num])
    for i in range(batch_size):
        if i%2 == 0:
            num = randint(1,11499)
            labels.append([1,0]) # 二分类标签表示，正例
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num - 1]
    return arr,labels

def get_test_batch():
    labels = []
    arr = np.zeros([batch_size,max_seq_num])
    for i in range(batch_size):
        num = randint(11499,13499)
        if num <= 12499:
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num - 1]
    return arr,labels


      
# 模型构建
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32,[batch_size,num_labels])
input_data = tf.placeholder(tf.int32,[batch_size,max_seq_num])

#data = tf.Variable(tf.zeros([batch_size,max_seq_num,num_dimensions]))#num_dimensions表示词向量的维数，此处为50 Dimensions for each word vector
data = tf.nn.embedding_lookup(word_vectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell,output_keep_prob=0.75)
value,_ = tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)

# dynamic_rnn的第一个输出可以被认为是最后的隐藏状态，该向量将重新确定维度，然后乘以一个权重加上bias,获得最终的label
weight = tf.Variable(tf.truncated_normal([lstm_units,num_labels]))
bias = tf.Variable(tf.constant(0.1,shape=[num_labels]))
value = tf.transpose(value,[1,0,2])
last = tf.gather(value,int(value.get_shape()[0])-1)
prediction = (tf.matmul(last,weight)+bias)

# 正确的预测函数以及正确率评估参数
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 最后将标准的交叉熵损失函数定义为损失值，选择Adam作为优化函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#使用Tensorboard可视化损失值和正确值


# 模型训练
import datetime

if not inference_mode:
    sess = tf.InteractiveSession()
    tf.device("/gpu:0")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    tf.summary.scalar('Loss',loss)
    tf.summary.scalar('Accuracy',accuracy)
    merged = tf.summary.merge_all()
    
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir,sess.graph)
    
    for i in range(iterations):
        # 下个批次的数据
        nextBatch,nextBatchLabels = get_train_batch();
        sess.run(optimizer,{input_data:nextBatch,labels:nextBatchLabels})
        
        # 每50次写入一次Tensorboard
        if i % 50 == 0:
            summary = sess.run(merged,{input_data:nextBatch,labels:nextBatchLabels})
            writer.add_summary(summary,i)
            
        # 每10，000次保存一下模型
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess,"models/pretrained_lstm.ckpt",global_step=i)
            print("saved to %s" % save_path)
        writer.close()
else:
# 预测部分
# 加载模型
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    iterations = 10
    for i in range(iterations):
      next_batch, next_batch_labels = get_test_batch()
      #
      print("正确率:", (sess.run(
          accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)