# *_* coding:utf-8 *_*
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
str_file = 'G://AgriculturalDisease_validation_annotations.json'


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


#
# def get_batch(image_id, label, batch_size):
#     rand = len(image_id)
#     image_batch = []
#     label_batch = []
#     for i in range(batch_size):
#         number = random.randint(0, rand)
#         image_batch.append(image_id[number])
#         label_batch.append(label[number])
#     return image_batch, label_batch



# train_dir = 'D:/picture/train/'
#
#
# def get_files(file_dir):
#     A5 = []
#     label_A5 = []
#     A6 = []
#     label_A6 = []
#     SEG = []
#     label_SEG = []
#     SUM = []
#     label_SUM = []
#     LTAX1 = []
#     label_LTAX1 = []
#
#     for file in os.listdir(file_dir):
#         name = file.split(sep='.')
#         if name[0] == 'A5':
#             A5.append(file_dir + file)
#             label_A5.append(0)
#         elif name[0] == 'A6':
#             A6.append(file_dir + file)
#             label_A6.append(1)
#         elif name[0] == 'LTAX1':
#             LTAX1.append(file_dir + file)
#             label_LTAX1.append(2)
#         elif name[0] == 'SEG':
#             SEG.append(file_dir + file)
#             label_SEG.append(3)
#         else:
#             SUM.append(file_dir + file)
#             label_SUM.append(4)
#
#     print('There are %d A5\nThere are %d A6\nThere are %d LTAX1\nThere are %d SEG\nThere are %d SUM' % (
#         len(A5), len(A6), len(LTAX1), len(SEG), len(SUM)))
#
#     image_list = np.hstack((A5, A6, LTAX1, SEG, SUM))
#     label_list = np.hstack((label_A5, label_A6, label_LTAX1, label_SEG, label_SUM))
#
#     temp = np.array([image_list, label_list])
#     temp = temp.transpose()
#     np.random.shuffle(temp)
#
#     image_list = list(temp[:, 0])
#     label_list = list(temp[:, 1])
#     label_list = [int(i) for i in label_list]
#
#     return image_list, label_list
#
#
# def get_batch(image, label, image_W, image_H, batch_size, capacity):
#     image = tf.cast(image, tf.string)
#     label = tf.cast(label, tf.int32)
#     # tf.cast()用来做类型转换
#
#     input_queue = tf.train.slice_input_producer([image, label])
#     # 加入队列
#
#     label = input_queue[1]
#     image_contents = tf.read_file(input_queue[0])
#     image = tf.image.decode_jpeg(image_contents, channels=3)
#     # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
#
#     image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
#     # resize
#
#     image = tf.image.per_image_standardization(image)
#     # 对resize后的图片进行标准化处理
#
#     image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)
#
#     label_batch = tf.reshape(label_batch, [batch_size])
#     return image_batch, label_batch
#

def weight_variable(shape):
    # 截断正太分布，限制了正态分布的变量取值区间,截断正太分布的标准差为stddev=0.1
    # 生成一个shape形状的，标准差为0.1的ndarray
    initial = tf.truncated_normal(shape, stddev=0.1)
    # 将它设置为变量
    return tf.Variable(initial)


# 令shape的值都为0.1
def bias_variable(shape):
    # 先设置为一个常量，然后变为变量
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 第一个参数是输入的信息，是一个list:[批量,输入的高度,输入的宽度,输入的频道]
    # 频道是rgb频道
    # 第二个参数是过滤器 也是一个list:[过滤器的高度,过滤器的宽度,输入频道,输出频道]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2(x):
    # padding=same指的是越过了图片的取样的地方取0
    # ksize表示pool窗口大小为2x2,也就是高2,宽2,
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 256 * 256])
y_ = tf.placeholder("float", shape=[None, 61])

# 第1层，卷积层
# 初始化W为[5,5,1,32]的张量，
# 表示卷积核大小为5*5,输入层1张图，有32个filter
W_conv1 = weight_variable([5, 5, 1, 32])
# 1行32列
b_conv1 = bias_variable([32])

# 把输入x(二维张量,shape为[batch, 65536])变成2d的x_image
# x_image的shape应该是[batch,256,256,1],1是图片的通道,灰度图片通道为1,rgb为3
# -1表示自动推测这个维度的size
x_image = tf.reshape(x, [-1, 256, 256, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 32个28*28的FM
h_pool1 = max_pool_2(h_conv1)

# 第二层卷积，前一层32个输入
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2(h_conv2)

# 密集的全连接层，1024个输出,一张图片缩小到7*7,前一层层共计64个特征图，全部作为输入
W_fc1 = weight_variable([64 * 64 * 64, 1024])
b_fc1 = bias_variable([1024])

# 将前一层处理的h_pool2改成一张7*7，共计64张的输入图
h_pool2_flat = tf.reshape(h_pool2, [-1, 64 * 64 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，在输出层之前加入dropout,没有shape表示就一个值
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层，1024个输入，61个输出
W_fc2 = weight_variable([1024, 61])
b_fc2 = bias_variable([61])
# 10 outputs
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
# 使用ADAM优化器来做梯度下降
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
image_id, label = get_files(str_file)

for i in range(20000):
    batch = get_image(28, image_id, label)
    # batch[1] = np.asarray(batch[1], dtype=np.float)

    if i % 50 == 0:
        # keep_prob 用于控制dropout的比例
        # 只是评估而已，并没有执行
        # print("correction")
        # print(sess.run(y_conv, {x: np.reshape(batch[0],(-1,256*256)), y_: np.reshape(batch[1], (-1, 61)), keep_prob: 1.0}))
        #
        # print(sess.run(y_, {x: np.reshape(batch[0],(-1,256*256)), y_: np.reshape(batch[1], (-1, 61)), keep_prob: 1.0}))

        train_accuracy = accuracy.eval(
            feed_dict={x: np.reshape(batch[0],(-1,256*256)), y_: np.reshape(batch[1], (-1, 61)), keep_prob: 1.0})

        print("step %d,train_accuracy %g" % (i, train_accuracy))
    # 批量执行
    train_step.run(feed_dict={x: np.reshape(batch[0],(-1,256*256)), y_: np.reshape(batch[1], (-1, 61)), keep_prob: 1})

# # print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
