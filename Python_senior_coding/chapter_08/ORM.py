#!/usr/bin/pythona
# -*- coding:utf-8 -*-
# @Time    : 4/30/19 2:22 PM
# @Author  : zhanfengchen
# @Site    : 
# @File    : ORM.py
# @Software: PyCharm
"""
1. 每个类都有一个Metaclass。在构造的时候，会按照自己、父类的顺序去寻找Metaclass。
2. 每个类只需要构造一次就好，构造顺序和__init__()一样

3. __new__()只需要有一个就好

"""
from numbers import Integral


class OutofRangeException(Exception):
    pass


class ErrorValueException(Exception):
    pass


class Field(object):
    def __init__(self):
        pass


class IntField(Field):
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
        super(IntField, self).__init__()

    def __set__(self, instance, value):
        if not isinstance(value, Integral):
            raise TypeError("value must be Integer.")

        if not (self._min_value <= value < self._max_value):
            raise OutofRangeException("The value {value} should between {min} and {max}".format())

        self._value = value

    def __get__(self, instance, owner):
        return self._value

    def save(self):
        pass


class StringField(Field):
    def __init__(self, db_column=None, max_length=None):
        self._value = None
        self._db_column = db_column
        self._max_length = max_length

    def __set__(self, instance, value):
        self._value = value

    def __get__(self, instance, owner):
        return self._value


class ModelMetaClass(type):
    """ 取出聚合数据 """

    def __new__(cls, name, bases, attr, **kwargs):
        print "1", name, bases, attr, kwargs
        if name == "ModelBase":
            return super(ModelMetaClass, cls).__new__(cls, name, bases, attr, **kwargs)

        fields = {}
        db_table = None
        for k, value in attr.items():
            if isinstance(value, Field):
                fields[value._db_column] = value
        meta = attr.get('Meta', None)
        if meta:
            db_table = getattr(meta, "db_table", None)
        attr['db_table'] = db_table
        attr['fields'] = fields

        return super(ModelMetaClass, cls).__new__(cls, name, bases, attr, **kwargs)


class BaseModel(object):
    __metaclass__ = ModelMetaClass

    # def __new__(cls, *args, **kwargs):
        # print "2", args, kwargs
        # return super(BaseModel, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        print "22"
        for k, v in kwargs.items():
            setattr(self, k, v)
        super(BaseModel, self).__init__()


class Company(BaseModel):
    scale = IntField(**{"db_column": 'scale', "min_value": 10, "max_value": 100})
    name = StringField(**{"db_column": 'name', "max_length": 80})

    class Meta(object):
        db_table = "company"


class User(BaseModel):
    age = IntField(**{"db_column": 'age', "min_value": 10, "max_value": 100})
    name = StringField(**{"db_column": 'name', "max_length": 80})

    class Meta(object):
        db_table = "user"


if __name__ == '__main__':
    print "=========================="
    cc = User(name='lll', age=23)

    print cc.age, cc.name

    ccd = User()
    ccd.age = 23
    print ccd.age
    print cc.age
