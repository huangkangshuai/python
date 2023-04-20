import numpy as np
from matplotlib import pyplot as plt

# 定义坐标轴
ax1 = plt.axes(projection='3d')

z = np.linspace(0, 13, 1000)
x = 90 * np.sin(z)
y = 90 * np.cos(z)
ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
plt.show()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
# import random
# import cv2
#
#
# # 根据字母的形状, 将字母转化为多个随机点
# def get_masked_data(letter, intensity=2):
#     # 多个随机点填充字母
#     random.seed(420)
#     x = []
#     y = []
#
#     for i in range(intensity):
#         x = x + random.sample(range(0, 1000), 500)
#         y = y + random.sample(range(0, 1000), 500)
#
#     if letter == ' ':
#         return x, y
#
#     # 获取图片的mask
#     mask = cv2.imread(f'images/letters/{letter.upper()}.png', 0)
#     mask = cv2.flip(mask, 0)
#
#     # 检测点是否在mask中
#     result_x = []
#     result_y = []
#     for i in range(len(x)):
#         if (mask[y[i]][x[i]]) == 0:
#             result_x.append(x[i])
#             result_y.append(y[i])
#
#     # 返回x,y
#     return result_x, result_y
#
#
# # 将文字切割成一个个字母
# def text_to_data(txt, repeat=True, intensity=2):
#     print('将文本转换为数据\n')
#     letters = []
#     for i in txt.upper():
#         letters.append(get_masked_data(i, intensity=intensity))
#     # 如果repeat为1时,重复第一个字母
#     if repeat:
#         letters.append(get_masked_data(txt[0], intensity=intensity))
#     return letters
#
#
# def build_gif(coordinates_lists, gif_name='movie', n_frames=10, bg_color='#95A4AD',
#               marker_color='#283F4E', marker_size=25):
#     print('生成图表\n')
#     filenames = []
#     for index in np.arange(0, len(coordinates_lists) - 1):
#         # 获取当前图像及下一图像的x与y轴坐标值
#         x = coordinates_lists[index][0]
#         y = coordinates_lists[index][1]
#
#         x1 = coordinates_lists[index + 1][0]
#         y1 = coordinates_lists[index + 1][1]
#
#         # 查看两点差值
#         while len(x) < len(x1):
#             diff = len(x1) - len(x)
#             x = x + x[:diff]
#             y = y + y[:diff]
#
#         while len(x1) < len(x):
#             diff = len(x) - len(x1)
#             x1 = x1 + x1[:diff]
#             y1 = y1 + y1[:diff]
#
#         # 计算路径
#         x_path = np.array(x1) - np.array(x)
#         y_path = np.array(y1) - np.array(y)
#
#         for i in np.arange(0, n_frames + 1):
#             # 计算当前位置
#             x_temp = (x + (x_path / n_frames) * i)
#             y_temp = (y + (y_path / n_frames) * i)
#
#             # 绘制图表
#             fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
#             ax.set_facecolor(bg_color)
#             plt.xticks([])  # 去掉x轴
#             plt.yticks([])  # 去掉y轴
#             plt.axis('off')  # 去掉坐标轴
#
#             plt.scatter(x_temp, y_temp, c=marker_color, s=marker_size)
#
#             plt.xlim(0, 1000)
#             plt.ylim(0, 1000)
#
#             # 移除框线
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#
#             # 网格线
#             ax.set_axisbelow(True)
#             ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
#             ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
#
#             # 保存图片
#             filename = f'images/frame_{index}_{i}.png'
#
#             if (i == n_frames):
#                 for i in range(5):
#                     filenames.append(filename)
#
#             filenames.append(filename)
#
#             # 保存
#             plt.savefig(filename, dpi=96, facecolor=bg_color)
#             plt.close()
#     print('保存图表\n')
#     # 生成GIF
#     print('生成GIF\n')
#     with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
#         for filename in filenames:
#             image = imageio.imread(filename)
#             writer.append_data(image)
#     print('保存GIF\n')
#     print('删除图片\n')
#     # 删除图片
#     for filename in set(filenames):
#         os.remove(filename)
#
#     print('完成')
#
#
# coordinates_lists = text_to_data('Python', repeat=True, intensity=50)
#
# build_gif(coordinates_lists,
#           gif_name='Python',
#           n_frames=7,
#           bg_color='#52A9F0',
#           marker_color='#000000',
#           marker_size=0.2)