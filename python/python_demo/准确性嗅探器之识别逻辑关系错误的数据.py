# 特征预处理（preprocess_data函数）：该函数对输入的数据进行预处理，根据不同字段类型进行相应处理。
# 深度学习模型训练（train_model函数）：该函数使用多层感知器（MLP）作为深度学习模型，用于训练模型以识别字段间的逻辑关系。
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier


def preprocess_data(df):
    processed_data = pd.DataFrame()

    # 处理数值和日期字段
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    date_columns = df.select_dtypes(include=['datetime64']).columns

    for col1 in numeric_columns:
        for col2 in numeric_columns:
            if col1 != col2:
                diff = df[col1] - df[col2]
                processed_data[f'{col1}_minus_{col2}'] = np.where(diff < 0, 0, np.where(diff == 0, 1, 2))

    for col1 in date_columns:
        for col2 in date_columns:
            if col1 != col2:
                diff = (df[col1] - df[col2]).dt.total_seconds()
                processed_data[f'{col1}_minus_{col2}'] = np.where(diff < 0, 0, np.where(diff == 0, 1, 2))

    # 处理数值类文本字段和日期类文本字段
    numeric_text_columns = ['numeric_text_column1', 'numeric_text_column2', ...]
    date_text_columns = ['date_text_column1', 'date_text_column2', ...]

    for col1 in numeric_text_columns:
        for col2 in numeric_text_columns:
            if col1 != col2:
                diff = df[col1].fillna(0) - df[col2].fillna(0)
                processed_data[f'{col1}_minus_{col2}'] = np.where(diff < 0, 0, np.where(diff == 0, 1, 2))

    for col1 in date_text_columns:
        for col2 in date_text_columns:
            if col1 != col2:
                diff = pd.to_datetime(df[col1], errors='coerce') - pd.to_datetime(df[col2], errors='coerce')
                diff = diff.dt.total_seconds().fillna(0)
                processed_data[f'{col1}_minus_{col2}'] = np.where(diff < 0, 0, np.where(diff == 0, 1, 2))

    # 处理其他字段
    other_columns = df.columns.difference(numeric_columns).difference(date_columns) \
        .difference(numeric_text_columns).difference(date_text_columns)

    for col in other_columns:
        if col not in ['enum_column1', 'enum_column2', ...]:
            processed_data[col] = np.where(df[col].isna(), 0, 1)

    # 处理枚举字段
    enum_columns = ['enum_column1', 'enum_column2', ...]
    for col in enum_columns:
        processed_data[col] = np.where(df[col].isna(), 0, df[col].astype('category').cat.codes + 1)

    return processed_data


def train_model(X, y):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X, y)

    return model


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 假设已经有了标签字段的列表
label_column = 'label'

# 预处理数据
processed_data = preprocess_data(df)

# 提取特征和标签
X = processed_data.drop([label_column], axis=1)
y = processed_data[label_column]

# 训练模型
model = train_model(X, y)

# 保存模型
model.save('logic_relation_model.h5')
