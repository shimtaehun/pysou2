# 비선형회귀분석
# 선형관계분석의 경우 모델에 다항식 또는 교호작용이 있는 경우에는 해석이 덜 직관적이다.
# 결과의 신뢰성이 떨어진다.
# 선형가정이 어긋날 때(정규성 위배) 대처하는 방법으로 다항식 항을 추가한 다항회귀 모델을 작성할 수 있다.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
# plt.scatter(x, y)
# plt.show()
print(np.corrcoef(x, y))    # 0.4807

# 선형회귀; 모델 작성
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]    # 차원 확대
# print(x)
model1 = LinearRegression().fit(x, y)
ypred = model1.predict(x)
print('예측값: ', ypred)
print('결정계수1: ', r2_score(y, ypred))

plt.scatter(x, y)
plt.plot(x, ypred, c='red')
plt.show()

# 다항회귀 모델 작성 - 추세선의 유연성을 위해 열 추가
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False) # degree=열의 수
x2 = poly.fit_transform(x)  # 특징 행렬을 만듦
print(x2)
model2 = LinearRegression().fit(x2, y)
ypred2 = model2.predict(x2)
print('예측값2: ', ypred2)  # [4.14285714 1.62857143 1.25714286 3.02857143 6.94285714]
print('결정계수2: ', r2_score(y, ypred2))   # 0.9892183288409704
plt.scatter(x, y)
plt.plot(x, ypred2, c='blue')
plt.show()

