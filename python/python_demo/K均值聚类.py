from sklearn.cluster import KMeans
import numpy as np

# 生成数据集
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# 创建 KMeans 模型
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)

# 训练模型并执行聚类操作
kmeans.fit(X)

# 预测新的数据点所属的类别
new_points = np.array([[0, 0], [12, 3]])
predicted_labels = kmeans.predict(new_points)

print(predicted_labels)