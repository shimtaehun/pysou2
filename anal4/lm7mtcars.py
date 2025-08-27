# 선형회귀분석 - ols 사용
# mtcars dataset 사용 - 독립변수가 종속변수(mps, 연비)에 영향을 미치는가?

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
# print(mtcars)
print(mtcars.columns) # ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(mtcars.info())
# print(mtcars.corr())
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0, 1]) # -0.7761683718265864
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0, 1]) # -0.8676593765172281

# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('hp')
# plt.ylabel('mpg')
# plt.show()

print('\n단순선형회귀 --------')
result = smf.ols(formula = 'mpg ~ hp', data = mtcars).fit()
print(result.summary())

#                                       hp         Intercept 
print('마력수 110에 대한 연비 예측 : ', -0.0682 * 110 + 30.0989)
print('마력수 50에 대한 연비 예측 : ', -0.0682 * 50 + 30.0989)
print('마력수 110에 대한 연비 예측 : ', result.predict(pd.DataFrame({'hp':[110]})))
print('마력수 50에 대한 연비 예측 : ', result.predict(pd.DataFrame({'hp':[50]})))

print('다중선형회귀')
result2 = smf.ols(formula = 'mpg ~ hp + wt', data = mtcars).fit()
print(result2.summary())
