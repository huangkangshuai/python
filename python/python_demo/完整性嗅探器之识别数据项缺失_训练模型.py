# 读取CSV文件，并对数据进行去噪处理，即去除数据中的异常值。
#
# 对枚举字段进行处理，将枚举类转换为数字。
#
# 对非枚举字段进行向量化处理，将空值转换为0，非空值转换为1。
#
# 对每个有空值的字段训练一个MLP分类器，并存储在models字典中。
#
# 针对每个有空值的字段，预测其空值，用MLP分类器进行预测。
#
# 进行主成分分析，将数据降维。
#
# 计算每个主成分对总方差的贡献比例。
#
# 将处理后的数据保存到CSV文件中。
#
# 在训练流程中，主要使用了sklearn库中的MLPClassifier和PCA两个模块。同时还使用了pandas和numpy库对数据进行处理和计算。在训练过程中，针对有空值的字段进行了处理，使用了MLP分类器进行预测填充空值。最后，对数据进行主成分分析，以提高模型效率和精度。
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import joblib


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

# 训练模型并将其存储在models字典中
models = joblib.load('C:/Users/86185/Desktop/models.pkl')

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

# 将models存储
joblib.dump(models, 'C:/Users/86185/Desktop/models.pkl')
