# K均值聚类（K-Means Clustering）：通过将数据点分成不同的簇，找出数量极少且离群的数据点
# DBSCAN离群点检测（Density-Based Spatial Clustering of Applications with Noise）：通过密度聚类的方式，检测数据集中的离群点。
# 通过聚类和密度的概念来识别离群数值。K均值聚类基于数据点之间的距离进行分组，找出数量较少的簇作为离群簇。DBSCAN离群点检测则基于数据点的密度，将离群点与其他点区分开。
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor


def identify_outlier_values(df, numeric_columns):
    outlier_values = {}
    for column in numeric_columns:
        values = df[column].dropna().values.reshape(-1, 1)

        # 使用K均值聚类将数据分为两个簇
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(values)
        cluster_labels = kmeans.labels_

        # 根据聚类结果，找出数量极少且离群的数据点
        cluster_counts = np.bincount(cluster_labels)
        outlier_cluster = np.argmin(cluster_counts)
        outlier_mask = cluster_labels == outlier_cluster
        outlier_values[column] = values[outlier_mask]

    return outlier_values


def identify_outlier_values_dbscan(df, numeric_columns):
    outlier_values = {}
    for column in numeric_columns:
        values = df[column].dropna().values.reshape(-1, 1)

        # 使用DBSCAN离群点检测算法
        clf = LocalOutlierFactor()
        outlier_labels = clf.fit_predict(values)

        # 找出离群的数据点
        outlier_mask = outlier_labels == -1
        outlier_values[column] = values[outlier_mask]

    return outlier_values


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 假设已经有了数值类型字段的列表
numeric_columns = ['numeric_column1', 'numeric_column2', ...]

# 识别离群的数值
outlier_values = identify_outlier_values(df, numeric_columns)

# 输出离群数值的异常数据
for column, values in outlier_values.items():
    print(f"Column: {column}, Outlier Values: {values}")
