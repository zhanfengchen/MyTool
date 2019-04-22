# -*- coding:utf-8 -*-


class Environment(object):
    def __init__(self, W, H):
        self._A = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 按照顺时针，上、左、下、右移动。
        self._W = W
        self._H = H
        self._end_set = set([(0, 0), (W-1, H-1)])

    @property
    def nA(self):
        return len(self._A)

    @property
    def W(self):
        return self._W

    @property
    def H(self):
        return self._H

    def move(self, status, action):
        x, y = status
        x_off, y_off = self._A[action]
        next_status = (x + x_off, y + y_off)
        done = next_status in self._end_set
        return next_status, done
