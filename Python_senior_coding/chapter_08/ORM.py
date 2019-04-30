#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time    : 4/30/19 2:22 PM
# @Author  : zhanfengchen
# @Site    : 
# @File    : ORM.py
# @Software: PyCharm
from numbers import Integral

class OutofRangeException(Exception):
    pass


class ErrorValueException(Exception):
    pass


class IntField(object):
    def __init__(self, db_column=None, min_value=10, max_value=30):
        self._value = None
        self._db_column = db_column

        if min_value < 0:
            raise ErrorValueException("The min_value must be positive integer, not {value}".format(value=min_value))

        if max_value < 0:
            raise ErrorValueException("The max_value must be positive integer, not {value}".format(value=max_value))

        if max_value < min_value:
            raise ErrorValueException("The max_value {max} must be more than min_value {value}".format(max=max_value,
                                                                                                       min=min_value))

        self._min_value = min_value
        self._max_value = max_value

    def __set__(self, instance, value):
        if isinstance(value, Integral):
            raise TypeError("value must be Integer.")

        if not (self._min_value <= value < self._max_value):
            raise OutofRangeException("The value {value} should between {min} and {max}".format())

        self._value = value

    def __get__(self, instance, owner):
        return self._value

    def save(self):
        pass


class ModelMetaClass(type):
    def __new__(cls, name, bases, attr, **kwargs):

        for k, value in attr:
            pass

        return super(ModelMetaClass, cls).__new__(cls, name, bases, attr, **kwargs)


class AA(object):
    __metaclass__ = ModelMetaClass
    age = IntField(20, **{"db_column": 'age', "min_value": 10})


if __name__ == '__main__':
    cc = AA()
