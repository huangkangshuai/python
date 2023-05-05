import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
np.seterr(divide='ignore',invalid='ignore')
data = pd.read_csv('C:/Users/86185/Desktop/datatest.csv')

# 去除噪音
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# 枚举字段处理
for col in data.columns:
    if col.startswith('enum_'):
        data[col] = data[col].apply(lambda x: int(x.split('_')[-1]))

# 向量化处理
for col in data.columns:
    if col.startswith('enum_'):
        data[col] = data[col].apply(lambda x: x - 1 if x != '' else 0)
    else:
        data[col] = data[col].apply(lambda x: 0 if pd.isnull(x) else 1)

# 训练模型
models = {}
for col in data.columns:
    if col != 'label' and data[col].isnull().sum() > 0:
        clf = MLPClassifier(random_state=42, max_iter=1000)
        clf.fit(data.dropna(subset=[col]).drop('label', axis=1), data.dropna(subset=[col])['label'])
        models[col] = clf

# 预测空值
for col in models.keys():
    null_idx = data[data[col].isnull()].index
    if len(null_idx) > 0:
        preds = models[col].predict(data.loc[null_idx, :].drop('label', axis=1))
        data.loc[null_idx, col] = preds

# 主成分分析
pca = PCA(n_components=0.9)
pca.fit(data.drop('label', axis=1))
components = pca.transform(data.drop('label', axis=1))

# 计算explained_variance_ratio_
explained_variance_ = pca.explained_variance_
total_var = np.sum(explained_variance_)
if total_var == 0:
    explained_variance_ratio_ = np.zeros(len(pca.explained_variance_ratio_))
else:
    explained_variance_ratio_ = np.nan_to_num(explained_variance_ / total_var)
data.to_csv('C:/Users/86185/Desktop/processed_data.csv', index=False)
