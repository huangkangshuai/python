import pandas as pd
from sklearn.cluster import KMeans

def identify_outlier_lengths(df):
    outlier_lengths = {}
    for column in df.columns:
        # 仅处理文本和编码类型的字段
        if column in text_columns or column in encoding_columns:
            lengths = df[column].dropna().astype(str).apply(len)
            kmeans = KMeans(n_clusters=2)  # 将长度聚类为2个簇
            kmeans.fit(lengths.values.reshape(-1, 1))
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            # 找到距离簇中心最远的数据长度
            max_distance = max(abs(lengths - cluster_centers[0]), abs(lengths - cluster_centers[1]))
            outlier_lengths[column] = max_distance.idxmax()

    return outlier_lengths


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 假设已经有了文本类型字段和编码类型字段的列表
text_columns = ['text_column1', 'text_column2', ...]
encoding_columns = ['encoding_column1', 'encoding_column2', ...]

# 识别离群的文本和编码字段长度
outlier_lengths = identify_outlier_lengths(df)

# 输出离群长度的异常数据
for column, length in outlier_lengths.items():
    print(f"Column: {column}, Outlier Length: {length}")
