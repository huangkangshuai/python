import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 准备数据
iris = load_iris()
X = iris.data
y = iris.target
datax = pd.DataFrame(X)
datax.to_csv('C:/Users/86185/Desktop/iris_data.csv', index=False)
datay = pd.DataFrame(y)
datay.to_csv('C:/Users/86185/Desktop/iris_target.csv', index=False)
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(y_test[:5])
print(X_test[:5])

print(y_train[:5])
print(X_train[:5])
# 训练模型
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 评估模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# 如果准确率为1，则将模型保存到本地
if accuracy == 1:
    joblib.dump(clf, "C:/Users/86185/Desktop/perfect_model.joblib")