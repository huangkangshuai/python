# class Test:
#     a = 0
#
#     def __init__(self, a, b):
#         print(a, b)
#         self.a = a
#
#
# t = Test(1, 2)
# print(t.a)
# !/usr/bin/python

# class Vector:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     def __str__(self):
#         return 'Vector (%d, %d)' % (self.a, self.b)
#
#     def __add__(self, other):
#         return Vector(self.a + other.a, self.b + other.b)

# v1 = Vector(2, 10)
# v2 = Vector(5, -2)
# print(v1 + v2)
# !/usr/bin/python
# -*- coding: UTF-8 -*-

# class Runoob:
#     __site = "www.runoob.com"
#
#
# runoob = Runoob()
# print(runoob._Runoob__site)

import pymysql

mydb = pymysql.connect(
    host="localhost",
    user="root",
    passwd="123456",
    database="python"
)
cursor = mydb.cursor()

table1 = 'userinfo'

sql = """select * from {0}""".format(table1)

cursor.execute(sql)
list = []
data = ()
while isinstance(data, tuple):  # 循环遍历出data的数据
    data = cursor.fetchone()  # fetchone方法用于取出数据库中查询的单个数据
    if (data == None): break
    obj = {}
    obj['id'] = data[0]
    obj['name'] = data[1]
    obj['age'] = data[2]
    obj['major'] = data[3]
    obj['hobby'] = data[4]
    print(obj)
    list.append(obj)
print(list)

