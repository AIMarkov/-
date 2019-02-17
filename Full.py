# *_* coding:utf-8 *_*
#可以直接使用3通道彩色
import tensorflow as tf
import random
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import re

a = []
import json

# ["disease_class"]
image_id = []
label = []
str_file = 'AgriculturalDisease_validation_annotations.json'


def get_files(str_file):
    image_id = []
    label = []
    with open(str_file, 'r') as f:
        print("Load str file from {}".format(str_file))
        str1 = f.read()
        r = json.loads(str1)
    for i in range(len(r)):
        a = [0.0] * 61
        name = r[i]['image_id'].split(sep='.')
        r[i]['image_id'] = str(name[0])
        image_id.append(r[i]['image_id'])

        a[r[i]['disease_class']] = a[r[i]['disease_class']] + 1.0
        label.append(np.asarray(a, dtype=np.float))
        # label.append(r[i]['disease_class'])

    return image_id, label


savepath = "processed/"


def get_image(batch, image_id, label):
    image_batch = []
    label_batch = []
    rand = len(image_id)
    for i in range(batch):
        number = random.randint(0, rand)
        # image_batch.append(image_id[number - 1])
        imagepath = savepath + image_id[number - 1]
        im = Image.open(imagepath)
        imnumpy = np.asarray(im, dtype=np.float)
        image_batch.append(np.reshape(imnumpy, (-1, 256 * 256)))
        label_batch.append(label[number - 1])
    return image_batch, label_batch


def get_files(str_file):
    image_id = []
    label = []
    with open(str_file, 'r') as f:
        print("Load str file from {}".format(str_file))
        str1 = f.read()
        r = json.loads(str1)
    for i in range(len(r)):
        a = [0.0] * 61

        # name = r[i]['image_id'].split(sep='.')
        name = re.sub('.JPG', '', r[i]['image_id'])
        name = re.sub('.jpg', '', name)
        r[i]['image_id'] = str(name)
        image_id.append(r[i]['image_id'])

        a[r[i]['disease_class']] = a[r[i]['disease_class']] + 1
        label.append(a)
        # label.append(r[i]['disease_class'])

    return image_id, label


sess = tf.Session()

# 第一个None是表示批量的样本输入,x不是一个特定的值，而是一个占位符placeholder
x = tf.placeholder("float", shape=[None, 256*256])
# 第一个None表示批量的样本标签
y_ = tf.placeholder("float", shape=[None, 61])

#赋予tf.Variable不同的初值来创建不同的Variable
W1 = tf.Variable(tf.random_normal([256*256,200]))
W2 = tf.Variable(tf.random_normal([200,61]))
b1 = tf.Variable(tf.zeros([200]))
b2 = tf.Variable(tf.zeros([61]))

# 初始化变量
sess.run(tf.initialize_all_variables())

a1 = tf.sigmoid(tf.matmul(x, W1) + b1)
a2 = tf.matmul(a1,W2) + b2
y = tf.nn.softmax(a2)



# y_是标签给的，y是我们预测的
#首先，用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。
corss_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

#自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(corss_entropy)
correct_pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, "float"))

image_id, label = get_files(str_file)
#使用with关键字的时候，就可以在Session中直接执行operation.run()或tensor.eval()两个类型的命令
with sess.as_default():
    for i in range(4000):
        batch = get_image(1028, image_id, label)
        #循环的每个步骤中，我们都会随机抓取训练数据中的50个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
        train_step.run(feed_dict={x: np.reshape(batch[0],(-1,256*256)), y_: np.reshape(batch[1], (-1, 61))},session=sess)
        print(accuracy.eval(feed_dict={x: np.reshape(batch[0], (-1, 256 * 256)), y_: np.reshape(batch[1], (-1, 61))},
                            session=sess))
# correct_pre返回布尔值的list,argmax(y, 1)1是找的每一行的最大值下标，0是列,返回的都是List
correct_pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, "float"))
print(accuracy.eval(feed_dict={x: np.reshape(batch[0],(-1,256*256)), y_: np.reshape(batch[1], (-1, 61))},session=sess))
sess.close()

