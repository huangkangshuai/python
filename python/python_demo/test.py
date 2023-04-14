# # 导包
# import requests
#
# # step_1 : 指定url
# url = 'https://www.sogou.com/'
# # step_2 : 发起请求:
# # 使用get 方法发起get 请求， 该方法会返回一个响应对象。参数url 表示请求对应的url
# response = requests.get(url=url)
# # step_3 : 获取响应数据:
# # 通过调用响应对象的text 属性， 返回响应对象中存储的字符串形式的响应数据（ 页面源码数据）
# page_text = response.text
# # step_4 : 持久化存储
# with open('C:/Users/86185/Desktop/sogou.html', 'w', encoding='utf -8') as fp:
#     fp.write(page_text)
# print('爬取数据完毕！ ！ ！')
import sys

# list1 = [10, 20, 30, 40]
# print(tuple(list1))
# print(list(list1))
# print(set(list1))
# # 列表字典集合推导式
# list1 = []
# i = 0
# while i < 10:
#     list1.append(i)
#     i += 1
# print(list1)
#
# for i in range(10):
#     list1.append(i)
# print(list1)
#
# list1 = [(i, j) for i in range(1, 10) for j in range(1, 10)]
# print(list1)
#
# list2 = ['name', 'age', 'gender']
# list3 = ['Tom', 20, 'man']
# dirt = {i: i ** 2 for i in range(1, 6)}
# print(dirt)
# dirt = {list2[i]: list3[i] for i in range(len(list2))}
# print(dirt)
# print(len(list2))
# counts = {'1': 268, '2': 125, '3': 201}
# print({key: value for key, value in counts.items() if value > 200})
#
# list1 = [1,2,3,4,3]
# set = {i **2 for i in list1}
# print(set)
# def hsms(a, b):
#     return a * a + b * b
#
#
# def hsm(a, b):
#     """求和"""
#     hsms(a, b)
#     return hsms(a, b)
#
#
# print(hsm(4, 6))
#
# help(hsm)
#
# a = 100
#
#
# def testA():
#     print(a)
#
#
# def testB():
#     global a
#     a = 200
#     print(a)
#
#
# testA()
# testB()
# print(a)

# def x(*args):
#     print(args)
#
#
# x(1)
# x(1, 2, 3)

# def x(**kwargs):
#     print(kwargs)
#
#
# x(a =1,b=2)

# def x():
#     return 100, 200
#
#
# a, b = x()
# print(x())
# print(a)
# print(b)

# dict1 = {'name':'TOM','age':18}
# a,b = dict1
#
# print(a)
# print(b)
#
# print(dict1[a],dict1[b])
# print(dict1['name'])

# def test1(a):
#     print(a)
#     print(id(a))
#
#     a += 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
#     print(a)
#     print(id(a))
#
#
# test1(1)

# while True:
#     a = input('输入值')
#     if a == '1':
#         break

# info = []
# x = input('id')
# y = input('name')
# a = {'name': y, 'id': x}
# info.append(a)
# print(info)

# def digui(num):
#     if num == 1:
#         return 1
#     return digui(num-1)+num
#
# #最大递归深度
# print(digui(996))

# 匿名函数
# lambda a,b=10: a+b
# c = lambda a,b=10: a+b
# print(c(1))
# print(c)

# lambda *args: args
# c = lambda *args:args+args
# print(c(1,2,3,4))
# print(c)

# lambda **kwargs: kwargs
# c = lambda **kwargs:kwargs
# print(c(b = 4))
# print(c)

# fn1 = lambda a, b: a if a > b else b
# print(fn1(1000, 500))
# student = [{'name': 'Jack', 'age': 20},
#            {'name': 'Rose', 'age': 19},
#            {'name': 'Tom', 'age': 18}]
# student.sort(key=lambda i:i['name'],reverse=True)
# print(student)
# for i in student:
#     print(i['name'])

# print(abs(-10))
# print(round(1.2))
# print(round(1.9))
# print(round(1.4))
# print(round(1.5))

# def sum_num(a,b,f):
#     return f(a)+f(b)
#
# print(sum_num(-1,-3,abs))

# 扫描全部
# list1 = [1, 2, 3, 4]


# def fun(x):
#     return x ** 2
#
#
# print(list(map(fun, list1)))
import functools

# -*- encoding:utf-8 -*-
# from flask import Flask
#
# app = Flask(__name__)
# @app.route('/')
# def index():
#     return 'Index Page'
#
# if __name__ == '__main__':
#     app.run(debug=True,port=3389)

# 累计计算
# import functools
#
# list1 = [1, 2, 3, 4, 5]
#
#
# def add(a, b):
#     return a + b
#
#
# result = functools.reduce(add, list1)
# print(result)

# 过滤元素
# list1 = [1, 2, 3, 4]
#
#
# def fun(x):
#     return x % 2 == 0
#
#
# print(list(filter(fun, list1)))

#ctrl z ctrl shift z ctrl f ctrl r