# 회귀문제 모델을 이용하여 아래의 문제를 해결하시오. 수학점수를 종속변수로 하자.
# - 국어 점수를 입력하면 수학 점수 예측
# - 국어, 영어 점수를 입력하면 수학 점수 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as lin
import statsmodels.api as sm
from sklearn.datasets import make_regression
import statsmodels.formula.api as smf
from scipy import stats


data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
print(data.head(3))

# - 국어 점수를 입력하면 수학 점수 예측
result = smf.ols(formula = '수학 ~ 국어', data = data).fit()
print('검정 결과 : ', result.summary()) # 0.5871584576511681
print('결정계수 : ', result.rsquared)
print('pvalue : ', result.pvalues.iloc[1])

print('국어점수 50에 대한 수학 점수 예측 : ', 0.5705 * 50 + 32.1069)
print('국어점수 50에 대한 수학 점수 예측 : ', result.predict(pd.DataFrame({'국어':[50]})))

# - 국어, 영어 점수를 입력하면 수학 점수 예측
result2 = smf.ols(formula = '수학 ~ 국어 + 영어', data = data).fit()
print('검정 결과 : ', result2.summary())

print('국어 50점, 영어 60점 입력 시 예측 수학 점수 : ', result.predict(pd.DataFrame({'국어' : [55], '영어' : [65]})))

new_data = pd.DataFrame({'국어':[55], '영어':[65]})
pred = result2.predict(new_data)
print('국어 50점, 영어 60점 입력 시 예측 수학 점수:', pred[0])