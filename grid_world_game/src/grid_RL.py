# -*- coding:utf-8 -*-
import math
import numpy as np
from environment import Environment


def itertools(envr, status, actions, status_reward_mtx, discount):
    rewards = []
    for action in actions:
        next_status, action_prob, done = envr.move(status, action)
        rewards.append(action_prob * (status_reward_mtx[status] + discount*status_reward_mtx.get(next_status, 0.)))
    return rewards


def rl(envr, threshold=0.00001):
    pointers = envr.pointers
    actions = envr.actions
    V = {p: 0.0 for p in pointers}
    for p in envr.end_set:
        V[p] = 1.

    ##
    last_value = 0.0
    while True:
        for p in pointers:
            rewards = itertools(envr, p, actions, V, 1.0)
            max_rewards = np.max(rewards)
            V[p] = max_rewards
        if abs(last_value - sum(V.values())) < threshold:
            break
    return V

if __name__ == '__main__':
    envr = Environment(4, 4)
    V = rl(envr)
    for i in range(4):
        print "\n"
        for j in range(4):
            print V[(i, j)],