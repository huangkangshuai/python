# 1.聚类算法（K-Means）：用于对字段进行聚类，将相似的值归为一类。
# 2.数据预处理：使用StandardScaler对聚类结果进行标准化，以便更好地适应模型训练。
# 3.逻辑回归模型（Logistic Regression）：用于训练模型和进行预测。通过提取特征和标签，将数据集分为训练集和测试集，并使用逻辑回归模型进行训练和评估。
# 4.数据预处理函数（preprocess_data）：用于进行特征工程，例如将非数值型字段转换为数值型、进行标准化等处理。在示例代码中，该函数没有具体实现，需要根据实际情况进行处理。
# 5.训练数据集构建函数（create_training_data）：根据识别到的主键和非唯一值构建训练数据集，其中'is_unique'字段表示该字段是否为唯一的主键。
# 6.总体来说，该代码通过对字段进行聚类分析，结合逻辑回归模型进行训练和预测，以识别实际主键和非唯一值，并生成预测结果。
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def identify_primary_key(df):
    primary_key = None
    non_unique_values = []

    for column in df.columns:
        # 对字段进行GroupBy
        grouped = df.groupby(column)
        group_counts = grouped.size()

        # 对GroupBy结果进行聚类
        scaler = StandardScaler()
        normalized_counts = scaler.fit_transform(group_counts.values.reshape(-1, 1))
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(normalized_counts)

        # 如果聚类结果收敛于1，则认为该字段为主键，并识别非唯一的值
        if len(set(labels)) == 1:
            primary_key = column
            non_unique_values = group_counts[group_counts > 1].index.tolist()
            break

    return primary_key, non_unique_values


def preprocess_data(df):
    # 进行特征工程，将非数值型字段转换为数值型，进行标准化等处理
    processed_df = df  # 这里需要根据实际情况进行处理
    return processed_df


def train_model(df):
    # 准备训练数据集
    processed_df = preprocess_data(df)

    # 提取特征和标签
    X = processed_df.drop(['is_primary_key'], axis=1)  # 去除标签列
    y = processed_df['is_primary_key']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义并训练模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 在测试集上评估模型性能
    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

    return model


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 调用函数识别主键和非唯一值
primary_key, non_unique_values = identify_primary_key(df)

# 构建训练数据集
training_data = pd.DataFrame({
    'column_name': [primary_key],
    'is_unique': [0]
})

for column in non_unique_values:
    training_data = training_data.append({
        'column_name': column,
        'is_unique': 1
    }, ignore_index=True)

# 训练模型
model = train_model(training_data)

# 使用模型进行预测
prediction = model.predict(df.drop(['is_primary_key'], axis=1))

# 将预测结果添加到原始DataFrame
df['predicted_primary_key'] = prediction

# 输出预测结果
print(df)
