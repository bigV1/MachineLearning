
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 创建一个随机的数据集
rng = np.random.RandomState(7)

X = np.sort(5 * rng.rand(80, 1), axis=0)
# print('X=', X)
y = np.sin(X).ravel()
# print('y=', y)
y[::5] += 3 * (0.5 - rng.rand(16))
# print('yyy=', y)

# 拟合回归模型
# regr_1 = DecisionTreeRegressor(max_depth=2)
# 保持 max_depth=5 不变，增加 min_samples_leaf=6 的参数，效果进一步提升了
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_2 = DecisionTreeRegressor(min_samples_leaf=6)
# regr_1.fit(X, y)
regr_2.fit(X, y)
# regr_3.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
# y_3 = regr_3.predict(X_test)

# 绘制结果
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.plot(X_test, y_3, color="red", label="max_depth=3", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()