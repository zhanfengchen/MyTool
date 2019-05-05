# -*- coding:utf-8 -*-
import math
import numpy as np
from environment import Environment


def itertools(envr, status, actions, status_reward_mtx, discount):
    rewards = []
    for action in actions:
        next_status, action_prob, done = envr.move(status, action)
        next_status_reward = status_reward_mtx.get(next_status, 0. if status in envr.end_set else -2.)
        value = action_prob * (status_reward_mtx[status] + discount * next_status_reward)
        rewards.append((action, value))
    return rewards


def rl(envr, threshold=0.0000001):
    pointers = envr.pointers
    actions = envr.actions
    V = {p: -1.0 for p in pointers}
    for p in envr.end_set:
        V[p] = 0.0

    ##
    last_value = 0.0
    v_actions = {p: 0 for p in pointers}
    while True:
        for p in pointers:
            rewards = itertools(envr, p, actions, V, 1.0)
            ac, max_reward = max(rewards, key=lambda x: x[1])
            V[p] = max_reward
            v_actions[p] = ac
        if abs(last_value - sum(V.values())) < threshold:
            break
    return V, v_actions


if __name__ == '__main__':
    envr = Environment(4, 4)
    V, ac = rl(envr)
    for i in range(4):
        print "\n"
        for j in range(4):
            print V[(i, j)],

    dir_mapping = dict(zip([(-1, 0), (0, 1), (1, 0), (0, -1)], [u"上", u"左", u"下", u"右", ]))
    for i in range(4):
        print "\n"
        for j in range(4):
            print dir_mapping[ac[(i, j)]],
