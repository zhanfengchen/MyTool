# -*- coding:utf-8 -*-


class Environment(object):
    def __init__(self, W, H):
        self._W = W
        self._H = H
        self._end_set = set([(0, 0), (W - 1, H - 1)])

        self._pointers = [(x, y)
                          for x in range(W)
                          for y in range(H)]

        self._actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 按照顺时针，上、左、下、右移动。
        self._action_prob_mapping = {p: {a: 1.0 / len(self._actions)
                                         for a in self._actions}
                                     for p in self._pointers}
    @property
    def end_set(self):
        return self._end_set

    @property
    def actions(self):
        return self._actions

    @property
    def pointers(self):
        return self._pointers

    @property
    def nA(self):
        return len(self._actions)

    @property
    def W(self):
        return self._W

    @property
    def H(self):
        return self._H

    def move(self, status, action):
        x, y = status
        x_off, y_off = action
        next_status = (x + x_off, y + y_off)
        done = next_status in self._end_set

        action_prob = self._action_prob_mapping.get(status, {}).get(action, 0.)
        return next_status, action_prob, done
