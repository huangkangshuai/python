import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取数据集
np.seterr(divide='ignore', invalid='ignore')
data = pd.read_csv('C:/Users/86185/Desktop/数据集/空值异常数据集/空值正常训练数据.csv', encoding='gbk')

# 去除噪音
numeric_cols = data.select_dtypes(include='number').columns
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
threshold = 3  # 调整离群值的阈值
data = data[~((data[numeric_cols] < (Q1 - threshold * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# 向量化处理
for col in data.columns:
    # 处理枚举字段
    if data[col].dtype == 'object':
        group_counts = data.groupby(col)[col].nunique()
    else:
        group_counts = data.groupby(col).nunique()

    if len(group_counts) < 10:
        enum_map = {np.nan: 0}  # 空值对应0
        enum_values = group_counts.index.values
        enum_classes = min(len(enum_values), 9)  # 最多映射为9个不同的值
        for i, value in enumerate(enum_values[:enum_classes]):
            enum_map[value] = i + 1  # 其他枚举类依次编码为1-9
        data[col] = data[col].map(enum_map).fillna(0).astype(int)
    else:
        data[col] = data[col].notna().astype(int)

# 创建空字典用于存储模型
models = {}

result_table = pd.DataFrame()

for ii in data.columns:
    # 划分特征和目标变量
    X = data.drop(ii, axis=1).values
    y = data[ii]

    # 创建训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 特征重要性排序
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': data.drop(ii, axis=1).columns, 'Importance': importance_scores})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # 选择前N个重要特征
    num_features = 5  # 选择前5个重要特征
    selected_features = feature_importance.head(num_features)['Feature'].values

    # 重新划分特征和目标变量
    X = data[selected_features].values

    # 创建训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Target Field: {ii}, Accuracy: {accuracy}')

    # 如果准确率大于0，则将模型保存到本地
    if accuracy > 0:
        models[ii] = model
        joblib.dump(model, f"C:/Users/86185/Desktop/数据集/空值异常数据集/{ii}_perfect_model.joblib")

        # 加载模型
        model = joblib.load(f'C:/Users/86185/Desktop/数据集/空值异常数据集/{ii}_perfect_model.joblib')

        # 读取数据集
        np.seterr(divide='ignore', invalid='ignore')
        dataX = pd.read_csv('C:/Users/86185/Desktop/数据集/空值异常数据集/空值异常测试数据.csv', encoding='gbk')
        # 向量化处理
        for col in dataX.columns:
            # 处理枚举字段
            if dataX[col].dtype == 'object':
                group_counts = dataX.groupby(col)[col].nunique()
            else:
                group_counts = dataX.groupby(col).nunique()

            if len(group_counts) < 10:
                enum_map = {np.nan: 0}  # 空值对应0
                enum_values = group_counts.index.values
                enum_classes = min(len(enum_values), 9)  # 最多映射为9个不同的值
                for i, value in enumerate(enum_values[:enum_classes]):
                    enum_map[value] = i + 1  # 其他枚举类依次编码为1-9
                dataX[col] = dataX[col].map(enum_map).fillna(0).astype(int)
            else:
                dataX[col] = dataX[col].notna().astype(int)

        # 提取特征列
        X = dataX[selected_features].values

        # 进行预测
        predictions = model.predict(X)

        # 将预测结果添加到测试数据集中
        df_predictions = pd.DataFrame({f'{ii}': predictions})
        result_table = pd.concat([result_table, df_predictions], axis=1)
result_table.to_csv(f'C:/Users/86185/Desktop/数据集/空值异常数据集/predictions.csv', index=False)
print(f'预测结果输出predictions.csv')
