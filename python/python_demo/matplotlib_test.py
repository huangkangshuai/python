import numpy as np
from matplotlib import pyplot as plt

# 定义坐标轴
ax1 = plt.axes(projection='3d')

z = np.linspace(0, 13, 1000)
x = 90 * np.sin(z)
y = 90 * np.cos(z)
ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
plt.show()
