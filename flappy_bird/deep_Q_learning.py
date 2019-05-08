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
import os
from game.wrapped_flappy_bird import GameState
import random

INIT_EPLISON = 0.1
FINAL_EPLISON = 0.00001
X_MEMORY = 50000
observe = 10000

ITERATION = 5000000

BATCH_SIZE = 32
ACTIONS = 2
GAMMA = 1.0

CHECK_POINTER_TIMES = 50000
MODEL_PATH = "./saved_networks"


def weight_variable(shape):
    variable = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(variable)


def bias_variable(shape):
    variable = tf.constant(0.01, shape=shape)
    return tf.Variable(variable)


def conv2d(input, filter, stride):
    return tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network():
    s_t = tf.placeholder("float", [None, 80, 80, 4], name="s_t")

    w_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    w_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    # 1st conv and pool
    h_conv1 = tf.nn.relu(conv2d(s_t, w_conv1, stride=4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2ed conv and pool
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, stride=2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 3th conv and pool
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, stride=1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # reshape
    h_pool = tf.reshape(h_pool3, shape=[-1, 256])

    # 1st full connection layer
    w_fc1 = weight_variable([256, 256])
    b_fc1 = bias_variable([256])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool, w_fc1) + b_fc1)

    # readout
    w_fc2 = weight_variable([256, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    readout = tf.matmul(h_fc1, w_fc2) + b_fc2

    return s_t, readout, w_fc1


def binarize_img(img):
    img80 = cv2.resize(img, (80, 80))
    img80_gray = cv2.cvtColor(img80, cv2.COLOR_RGB2GRAY)
    _, binarized_img = cv2.threshold(img80_gray, 1, 255, cv2.THRESH_BINARY)
    return binarized_img


def train(s_t, readout, w_fc, sess):
    # train
    a = tf.placeholder("float", shape=[None, ACTIONS], name='a')
    y = tf.placeholder("float", shape=[None], name="y")
    readout_tensor = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_tensor))
    train_step = tf.train.AdadeltaOptimizer(1e-6).minimize(cost)

    # stay guowangshuju
    D = deque(maxlen=X_MEMORY)

    # start with donoting
    game_state = GameState()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    next_img, reward, terminal = game_state.frame_step(do_nothing)
    b_img = binarize_img(next_img)
    tmp_x_sequ = np.stack([b_img, b_img, b_img, b_img], axis=2)  #

    time = 0
    # save model
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    check_pointer = tf.train.get_checkpoint_state(MODEL_PATH)
    if check_pointer and check_pointer.model_checkpoint_path:
        saver.restore(sess, check_pointer.model_checkpoint_path)
        import re
        match = re.search(u"\d+", check_pointer.model_checkpoint_path)
        if match:
            time = int(match.group())
        print("Load netword model successfully!")
    else:
        print("Could not find a network weights!")

    OBSERVE = time + observe
    while "bird fly":  # 小鸟起飞
        eplison = INIT_EPLISON * (ITERATION - time) / ITERATION + FINAL_EPLISON
        develop_ = ""
        _pred_readout = readout.eval(feed_dict={s_t: [tmp_x_sequ]})[0]
        action = np.zeros(ACTIONS)
        if random.random() <= eplison:  # 探索
            idx = random.randrange(ACTIONS)
            action[idx] = 1
            develop = u"探索"
        else:
            idx = np.argmax(_pred_readout)
            action[idx] = 1
            develop = u"开发"

        next_img, reward, terminal = game_state.frame_step(action)
        next_b_img = binarize_img(next_img)
        next_img_1 = np.reshape(next_b_img, (80, 80, 1))
        next_x_sequ = np.append(next_img_1, tmp_x_sequ[:, :, :3], axis=2)
        D.append((tmp_x_sequ, action, reward, next_x_sequ, terminal))

        if time > OBSERVE:  # train
            minibatch = random.sample(D, BATCH_SIZE)
            s_t_batch = [e[0] for e in minibatch]
            action_batch = [e[1] for e in minibatch]
            r_batch = [e[2] for e in minibatch]
            next_s_t_X = [e[3] for e in minibatch]

            y_batch = []
            readout_next_s_t_batch = readout.eval(feed_dict={s_t: next_s_t_X})
            for i, d in enumerate(minibatch):
                _terminal = d[4]
                if _terminal:
                    y_batch.append(r_batch[i] * 1.0)
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_next_s_t_batch[i]))

            train_step.run(feed_dict={
                y: y_batch,
                a: action_batch,
                s_t: s_t_batch
            })

        tmp_x_sequ = next_x_sequ

        if time and time % CHECK_POINTER_TIMES == 0:
            saver.save(sess, os.path.join(MODEL_PATH, "dqn-"), global_step=time)

        if time % 1 == 0:
            print(u"time:{time} / "
                  u"{develop} / "
                  u"eplison: {eplison} / "
                  u"reward:{reward} / {action} /"
                  u"Terminal:{terminal}/ q_max:{pred_readout}".format(time=time, develop=develop,
                                                eplison=eplison, reward=reward,
                                                action=("down" if action[0] > 0.9 else "up  "), terminal=terminal,
                                                pred_readout=_pred_readout))
        time += 1


def play_game():
    sess = tf.InteractiveSession()
    s_t, readout, w_fc = create_network()
    train(s_t, readout, w_fc, sess)


if __name__ == '__main__':
    play_game()
