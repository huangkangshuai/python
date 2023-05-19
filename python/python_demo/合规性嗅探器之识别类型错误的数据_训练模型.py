# SpaCy: 使用SpaCy的英文模型进行文本类型的识别。通过加载SpaCy的模型，对文本字段进行处理和判断是否为文本类型。
# CountVectorizer: 用于将文本数据进行向量化，将文本类型的字段转换为特征向量。通过对文本进行词频统计，生成特征矩阵。
# StandardScaler: 用于数值类型字段的标准化。通过对数值数据进行标准化处理，将其转换为均值为0，方差为1的数据。
# LogisticRegression: 用于训练分类模型。使用逻辑回归算法训练模型，将提取的特征数据用于预测数据类型。
# 该代码的作用是根据给定的数据表，自动识别字段的数据类型，并将其转换为数值化的特征表示。通过文本类型的识别、编码类型的识别和数值类型的判断，确定字段的数据类型。然后，对文本类型字段进行CountVectorizer向量化，对数值类型字段进行标准化处理，将编码类型字段转换为字符串形式。接着，使用逻辑回归算法训练模型，将字段的数据作为特征进行训练，并对字段的数据类型进行分类。最后，保存训练好的模型和数据类型信息，以供后续使用。该代码实现了对文本类字段进行智能分析和数据类型的自动标记。
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# 加载spaCy的英文模型
nlp = spacy.load('en_core_web_sm')

def identify_data_types(df):
    data_types = {}

    for column in df.columns:
        # 识别文本类型
        is_text = False
        for value in df[column].dropna():
            doc = nlp(value)
            if len(doc) > 0:
                is_text = True
                break
        if is_text:
            data_types[column] = 'Text'
            continue

        # 识别日期类型
        # 这部分需要根据具体情况自行实现

        # 识别编码类型
        is_encoding = False
        for value in df[column].dropna():
            if str(value)[0] == '0' or not str(value).isdigit():
                is_encoding = True
                break
        if is_encoding:
            data_types[column] = 'Encoding'
            continue

        # 数值类型
        data_types[column] = 'Numeric'

    return data_types


def preprocess_data(df, data_types):
    processed_data = pd.DataFrame()

    for column, data_type in data_types.items():
        if data_type == 'Text':
            vectorizer = CountVectorizer()
            text_features = vectorizer.fit_transform(df[column].fillna(''))
            text_feature_names = vectorizer.get_feature_names()
            for i in range(len(text_feature_names)):
                feature_name = f"{column}_{text_feature_names[i]}"
                processed_data[feature_name] = text_features[:, i].toarray().ravel()
        elif data_type == 'Numeric':
            scaler = StandardScaler()
            numeric_features = scaler.fit_transform(df[column].fillna(0).values.reshape(-1, 1))
            processed_data[column] = numeric_features.ravel()
        elif data_type == 'Encoding':
            processed_data[column] = df[column].fillna('').astype(str)

    return processed_data


def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)

    return model


# 读取表数据到DataFrame
df = pd.read_csv('your_table.csv')

# 识别数据类型
data_types = identify_data_types(df)

# 构建训练数据集
training_data = pd.DataFrame(columns=['column_name', 'data_type'])

for column, data_type in data_types.items():
    training_data = training_data.append({
        'column_name': column,
        'data_type': data_type
    }, ignore_index=True)

# 预处理数据
processed_data = preprocess_data(df, data_types)

# 提取特征和标签
X = processed_data.drop(['data_type'], axis=1)
y = processed_data['data_type']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = train_model(X_train, y_train)

# 在测试集上评估模型性能
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 保存模型
with open('data_type_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 保存数据类型信息
training_data.to_csv('data_types.csv', index=False)
