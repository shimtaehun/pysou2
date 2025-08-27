# 단순선형회귀: ols 사용
# 상관관계가 선형회귀 모델에 미치는 영향

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head(2))
print(iris.iloc[: , 0:4].corr())

# 연습 1: 상관관계가 약한(-0.117570) 두 변수(sepal_width, sepal_length)를 사용
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('검정 결과1 ', result1.summary())
print('결정계수 ', result1.rsquared)    # 0.013822654141080859
print('pvalue ', result1.pvalues[1])   #  0.15189826071144785 > 0.05    유의하지 않은 모델
# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color='r')

# 연습 2: 상관관계가 강한(0.871754) 두 변수(petal_widtj, sepal_length)를 사용
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
print('검정 결과2 ', result2.summary())
print('결정계수 ', result2.rsquared)    # 0.
print('pvalue ', result2.pvalues.iloc[1])
plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), color='r')
plt.show()
print('-'*100)
# 일부의 실제값과 예측값 비교
print('실제값: ', iris.sepal_length[:5].values)
print('예측값: ',  result2.predict()[:5])
print('-'*100)

# 새로운 값으로 예측
new_data = pd.DataFrame({'petal_length':[1.1, 0.5, 5.0]})
y_pred = result2.predict(new_data)
print('예측 결과(sepal_length):\n ', y_pred)

print('-- 다중 선형회귀: 복수--')
# result3 = smf.ols(formula='sepal_length ~ petal_length+petal_width+sepal_width', data = iris).fit()
column_select = "+".join(iris.columns.difference(['sepal_length', 'species']))
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data = iris).fit()
print(result3.summary())