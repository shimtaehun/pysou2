# 선형회귀분석 - ols 사용
# ----선형회귀분석 조건----
# 선형성 (Linearity)
# 독립변수(X)와 종속변수(Y) 사이 관계가 직선 형태.

# 독립성 (Independence)
# 오차항(ε)들이 서로 독립.

# 등분산성 (Homoscedasticity)
# 오차항의 분산이 모든 독립변수 값에서 일정.

# 정규성 (Normality)
# 오차항이 정규분포를 따름.

# 다중공선성 없음 (Multicollinearity)
# 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm # Quantile - Quantile plot 지원
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family = 'malgun gothic')

# 각 매체의 광고비에 따른 판매량 관련성
advdf = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv', usecols = [1,2,3,4])
print(advdf.head(3))
print(advdf.shape) # (200, 4)
print(advdf.index, ' ', advdf.columns)
print(advdf.info())
print(advdf.corr())

print()

# 단순선형회귀 모델 : x=tv, y=sales
lmodel1 = smf.ols(formula = 'sales ~ tv', data = advdf).fit()
print(lmodel1.params)
print(lmodel1.pvalues)
print(lmodel1.rsquared)
# print(lmodel1.summary())
print(lmodel1.summary().tables[0]) # Regression 윗부분만 출력

print('예측 --------------------')
x_part = pd.DataFrame({'tv' : advdf.tv[:3]})
print('실제값 : ', advdf.tv[:3])
print('예측값 : ', lmodel1.predict(x_part).values)
print('새 자료로 예측하기-----------')
x_new = pd.DataFrame({'tv' : [100,300,500]})
print('새 자료 예측값 : ', lmodel1.predict(x_new).values)

# 시각화
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('tv')
plt.ylabel('sales')
y_pred = lmodel1.predict(advdf.tv)
plt.plot(advdf.tv, y_pred, c = 'red') # 추세선
# plt.grid()
# plt.show()

print('*******선형회귀분석의 기본 충족 조건 ******************')

# 잔차항(예측값 - 실제값)
fitted = lmodel1.predict(advdf) # 예측값
# print('예측값 : \n', fitted)
residual = advdf['sales'] - fitted
print('실제값 : ', advdf['sales'][:5].values) # [22.1 10.4  9.3 18.5 12.9]
print('예측값 : ', fitted[:5].values) # [17.97077451  9.14797405  7.85022376 14.23439457 15.62721814]
print('잔차값 : ', residual[:5].values) # [ 4.12922549  1.25202595  1.44977624  4.26560543 -2.72721814]
print('잔차의 평균값 : ', np.mean(residual)) # -1.42e-15

print('정규성 : 잔차가 정규성을 따르는지?')
from scipy.stats import shapiro

stat, pv = shapiro(residual)
print(f"shapiro-wilk test → 통계량 : {stat:.4f}, p-value : {pv:.4f}") # 통계량 : 0.9905, p-value : 0.2133
print('정규성 만족' if pv > 0.05 else '정규성 위배 가능성↑')

# 시각화로 확인 - Q-Q plot
sm.qqplot(residual, line = 's')
plt.title('잔차 Q-Q plot')
plt.show() # 정규성이 만족하나 부분적 곡선이 존재

print('2) 선형성 : 독립변수의 변화에 따라 종속변수도 일정 크기로 변화해야한다')
from statsmodels.stats.diagnostic import linear_reset # 모형 적합성 확인
reset_result = linear_reset(lmodel1, power = 2, use_f = True)
print(f'linear_reset test : F = {reset_result.fvalue : .4f}, p = {reset_result.pvalue : .4f}') # F =  3.7036, p =  0.0557
print('선형성 만족' if reset_result.pvalue > 0.05 else '선형성 위배 가능성↑')

# 시각화로 확인
sns.regplot(x = fitted, y = residual, lowess = True, line_kws = {'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color = 'gray')
plt.show()

print('3) 독립성 : 독립변수의 값이 서로 관련 x')
# 독립성 가정은 잔차 간에 자기상관이 없어야 한다
# 자기상관 : 회귀분성 등에서 관측된 값과 추정된 값의 차이인 잔차들이 서로 연관되어있는 정도
# 듀빈 - 왓슨 검정으로 확인
print(lmodel1.summary()) #  Durbin-Watson : 1.935 → 2에 가까울수록 자기상관 없음

# 참고 : Cook's distance
# 하나의 관측치가 전체 모델에 얼마나 영향을 주는지 수치화한 지표

from statsmodels.stats.outliers_influence import OLSInfluence

cd, _ = OLSInfluence(lmodel1).cooks_distance # 쿡의 거리값, indew

print(cd.sort_values(ascending = False).head(3))
# 인덱스 번째에 해당하는 원본 자료 확인
print(advdf.iloc[[35, 178, 25, 175, 131]])
# 해석 : 대체적으로 tv 광고비는 높으나 sales가 적음 - 모델이 예측하기 어려운 포인트

# Cook's distance 시각화
fig = sm.graphics.influence_plot(lmodel1, alpha = 0.05, criterion = 'cooks')

plt.show()