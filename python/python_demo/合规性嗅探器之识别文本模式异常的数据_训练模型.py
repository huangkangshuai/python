# 中文分词（jieba）：用于将中文文本进行分词，将文本划分为有意义的词语。
# 正则表达式（re）：用于根据正则模式匹配文本中的特定模式，例如判断数字、字母、特殊字符等。
# 多层感知机（MLP）：一种深度学习算法，用于训练文本模式识别模型。MLP通过多个神经网络层的连接和非线性激活函数来学习输入数据的复杂模式。
# 特征向量化：将文本模式转换为数值型特征向量表示，便于机器学习算法的处理和训练。
# 整体流程是先进行中文分词和模式判断，然后将文本模式向量化，并将其作为特征输入到MLP模型中进行训练。训练完成后，模型可以用于识别不符合文本模式的数据。
import pandas as pd
import jieba
import re
import numpy as np
from sklearn.neural_network import MLPClassifier


def tokenize_text(text):
    words = jieba.cut(text, cut_all=False)
    tokens = [word.strip() for word in words if word.strip()]
    return tokens


def identify_text_patterns(df, text_columns):
    text_patterns = {}
    for column in text_columns:
        patterns = set()
        for value in df[column].dropna():
            tokens = tokenize_text(value)
            pattern = []
            for token in tokens:
                if re.match(r'^[0-9]+$', token):  # 数字
                    pattern.append(1)
                elif re.match(r'^[a-zA-Z]+$', token):  # 字母
                    pattern.append(2)
                elif re.match(r'^[-]+$|^[—]+$', token):  # 特殊字符 -
                    pattern.append(3)
                # 添加更多的模式判断，例如地址、人名等的识别
                else:  # 名词或其他类型
                    pattern.append(5)
            patterns.add(tuple(pattern))
        text_patterns[column] = patterns

    return text_patterns


def vectorize_text_patterns(text_patterns):
    vectorized_patterns = {}
    for column, patterns in text_patterns.items():
        unique_patterns = list(patterns)
        pattern_mapping = {pattern: i + 1 for i, pattern in enumerate(unique_patterns)}
        vectorized_patterns[column] = pattern_mapping

    return vectorized_patterns


def preprocess_text_data(df, vectorized_patterns):
    processed_data = pd.DataFrame()

    for column, patterns in vectorized_patterns.items():
        for pattern, mapping in patterns.items():
            feature_name = f"{column}_pattern_{mapping}"
            processed_data[feature_name] = np.zeros(len(df))
            for i, value in enumerate(df[column].fillna('')):
                tokens = tokenize_text(value)
                pattern_vector = [mapping[token] if token in mapping else 0 for token in tokens]
                if tuple(pattern_vector) == pattern:
                    processed_data.loc[i, feature_name] = 1

    return processed_data


def train_model(X, y):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X, y)

    return model


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 假设已经有了文本类型字段和编码类型字段的列表
text_columns = ['text_column1', 'text_column2', ...]
encoding_columns = ['encoding_column1', 'encoding_column2', ...]

# 识别文本字段的模式
text_patterns = identify_text_patterns(df, text_columns)

# 向量化文本模式
vectorized_patterns = vectorize_text_patterns(text_patterns)

# 预处理数据
processed_data = preprocess_text_data(df, vectorized_patterns)

# 提取特征和标签
X = processed_data.drop(['label'], axis=1)
y = processed_data['label']

# 训练模型
model = train_model(X, y)

# 保存模型
model.save('text_pattern_model.h5')
