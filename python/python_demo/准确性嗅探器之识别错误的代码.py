# K-Means聚类算法：用于将枚举代码进行聚类分组，以便识别离群代码。K-Means算法根据代码值的相似性将其分配到不同的簇中。
# 标签编码（Label Encoding）：对枚举类型字段进行标签编码，将其转换为数值形式，以便进行聚类算法的输入。
# 深度学习模型（Sequential Model）：使用TensorFlow的Keras库构建了一个深度学习模型。该模型采用了多个密集连接层（Dense Layer）组成的前馈神经网络结构。
# 二元交叉熵损失函数（Binary Cross Entropy Loss）：作为模型的损失函数，用于衡量二分类任务的预测结果与实际标签之间的差异。
# Adam优化器（Adam Optimizer）：作为模型的优化器，用于自适应调节学习率，并帮助模型更快地收敛到最优解。
# 识别离群代码：通过聚类算法，对枚举类型字段的代码值进行聚类分组，找出数量极少的代码簇，将其识别为离群代码。
# 特征工程：将识别到的离群代码值作为新的特征列添加到原始数据中，用于提供额外的信息用于训练深度学习模型。
# 模型训练：使用深度学习模型对预处理后的数据进行训练，学习特征与标签之间的关系，以便对未知数据进行异常值的预测和识别。
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, concatenate

def identify_outlier_codes(df, enum_columns):
    outlier_codes = {}
    for column in enum_columns:
        values = df[column].dropna().values.reshape(-1, 1)

        # 使用K均值聚类将枚举代码进行分组
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(values)
        cluster_labels = kmeans.labels_

        # 统计每个代码类别的数量
        cluster_counts = np.bincount(cluster_labels)

        # 找出数量极少的代码类别（离群代码）
        outlier_clusters = np.where(cluster_counts < 10)[0]

        # 找出离群代码值
        outlier_values = []
        for cluster in outlier_clusters:
            outlier_mask = cluster_labels == cluster
            outlier_values.extend(values[outlier_mask])

        outlier_codes[column] = outlier_values

    return outlier_codes


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 假设已经有了枚举类型字段的列表
enum_columns = ['enum_column1', 'enum_column2', ...]

# 识别离群代码
outlier_codes = identify_outlier_codes(df, enum_columns)

# 将离群代码值添加为新的特征列
for column, values in outlier_codes.items():
    df[f'{column}_outlier'] = np.where(df[column].isin(values), 1, 0)

# 预处理数据
# ...（根据需要进行数据预处理的步骤，如填充缺失值、标准化等）

# 提取特征和标签
X = df.drop(enum_columns, axis=1)  # 包含离群代码特征列的输入特征
y = df['label_column']  # 标签列

# 构建深度学习模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# 训练模型
model.fit(X, y, epochs=100, batch_size=32, verbose=0)

# 保存模型
model.save('outlier_code_model.h5')
