import joblib
from sklearn.datasets import load_iris


# 加载模型
model = joblib.load('C:/Users/86185/Desktop/perfect_model.joblib')

# 需要预测的数据
iris = load_iris()
data = iris.data
print(data)
# 进行预测
predictions = model.predict(data)

print(predictions)
