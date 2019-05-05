#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time    : 5/5/19 9:37 AM
# @Author  : zhanfengchen
# @Site    : 
# @File    : deep_Q_learning.py
# @Software: PyCharm
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import sys

sys.path.append("./game")
from wrapped_flappy_bird import GameState
import random

init_eplison = 0.1
final_eplison = 0.00001
X_memory = 50000
observe = 1000


batch_size = 64
ACTIONS = 2


def weight_variable(shape):
    variable = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(variable)


def bias_variable(shape):
    variable = tf.constant(0, shape=shape)
    return tf.Variable(variable)


def conv2d(input, filter, stride):
    return tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network():
    s_t = tf.placeholder(float, [None, 80, 80, 4], name="s_t")

    w_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    w_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    # 1st conv and pool
    h_conv1 = tf.nn.relu(conv2d(s_t, w_conv1, strip=4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2ed conv and pool
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, strip=2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 3th conv and pool
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, strip=1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # reshape
    h_pool = tf.reshape(h_pool3, shape=[-1, 256])

    # 1st full connection layer
    w_fc1 = weight_variable([256, 30])
    b_fc1 = bias_variable([30])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool, w_fc1) + b_fc1)

    # readout
    w_fc2 = weight_variable([30, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    readout = tf.matmul(h_fc1, w_fc2) + b_fc2

    return s_t, readout, w_fc1


def binarize_img(img):
    img80 = cv2.resize(img, (80, 80))
    img80_gray = cv2.cvtColor(img80, cv2.COLOR_RGB2GRAY)
    binarized_img = cv2.threshold(img80_gray, 1, 255, cv2.THRESH_BINARY)
    return binarized_img


def train(s_t, readout, w_fc, sess):
    # train
    a = tf.placeholder(float, shape=[None, ACTIONS], name='A')
    y = tf.placeholder(float, shape=[None], name="Y")
    readout_tensor = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_tensor))
    train_step = tf.train.AdadeltaOptimizer(0.00001).minimize(cost)

    # stay guowangshuju
    D = deque(maxlen=X_memory)

    # start with donoting
    game_state = GameState()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    status_img, reward, terminal = game_state.frame_step(do_nothing)
    b_img = binarize_img(status_img)
    tmp_x_sequ = np.stack([b_img, b_img, b_img, b_img], axis=2) #

    # save model
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    check_pointer = tf.train.get_checkpoint_state("./saved_networks")
    if check_pointer and check_pointer.model_checkpoint_path:
        saver.save(sess, check_pointer.model_checkpoint_path)
        print("Save netword model successfully!")
    else:
        print("Could not find old network weights!")


def play_game():
    sess = tf.InteractiveSession()
    s_t, readout, w_fc = create_network()
    train(s_t, readout, w_fc, sess)


if __name__ == '__main__':
    play_game()
