import numpy as np
from numpy import *
import matplotlib.pyplot as plt

epsilon1 = 1
epsilon2 = 1
epsilon_total = 2


# 满足1-差分隐私
def F1():
    return np.random.laplace(loc=0, scale=1 / epsilon1)


# 满足1-差分隐私
def F2():
    return np.random.laplace(loc=0, scale=1 / epsilon2)


# 满足2-差分隐私
def F3():
    return np.random.laplace(loc=0, scale=1 / epsilon_total)


# 根据串行组合性，满足2-差分隐私
def F_combined():
    return (F1() + F2()) / 2


# # 绘制F1
# plt.hist([F1() for i in range(1000)], bins=50, label='F1')
#
# # 绘制F2（看起来应该与F1相同）
# plt.hist([F2() for i in range(1000)], bins=50, alpha=.7, label='F2')
# plt.legend()
# plt.show()

# x = np.arange(1,11)
# y =  2  * x +  5
# plt.title("Matplotlib demo")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(x,y)
# plt.show()


# a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27, 120, 110])
# plt.hist(a, bins=[0, 20, 40, 60, 80, 100, 120])
# plt.title("histogram")
# plt.show()

print(eye(4))