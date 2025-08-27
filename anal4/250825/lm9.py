# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.
# ols 사용 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')
import seaborn 
import statsmodels.formula.api as smf
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv')
print(df.head(2))
print(df.info())
df = df.drop([df.columns[6],df.columns[9],df.columns[10]], axis=1)
print(df.corr())    # 종속변수 = sales     독립변수  = Income, Advertising, Price, Age
lmodel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=df).fit()
print('요약결과 : ', lmodel.summary())
# Income, Advertising, Price, Age 모두 < 0.05
# 작성된 모델 저장 후 읽어서 사용 
''''
# pickle 모듈 사용
import pickle   
# 저장
with open('mymodel.pickle', mode = 'wb') as obj:
    pickle.dump(lmodel, obj)
# 읽기
with open('mymodel.pickle', mode = 'rb') as obj:
    mymodel = pickle.load(obj)
mymodel.predict('~~~')
'''
'''
# joblib 모듈 사용
import joblib
# 저장
joblib.dump(lmodel, 'mymodel.model')
# 읽기
mymodel = joblib.load('mymodel.model')
mymodel.predict('~~~')
'''

# ***선형회귀분석의 기본 충족 조건***
print(df.head(3))
df_lm = df.iloc[:,[0,2,3,5,6]] # Sales, Income, Advertising, Price, Age만 남기기
# 잔차항 구하기
fitted = lmodel.predict(df_lm)
residual = df_lm['Sales'] - fitted
print(residual[3])
print('잔차의 평균 : ', np.mean(residual))

print('\n선형성 : 잔차가 일정하게 분포되어야 함')
# 시각화로 확인
sns.regplot(x = fitted, y = residual, lowess = True, line_kws = {'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color = 'gray')
plt.show() # 선형성 만족

print('\n정규성 : 잔차항이 정규분포를 따라야 함')
import scipy.stats as stats
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x=x, y=y)
plt.plot([3,-3], [-3,3], '--', color = 'gray')
plt.show()
print()
print('shapiro test : ', stats.shapiro(residual)) # statistic = 0.9949, pvalue = 0.2127 > 0.05 → 정규성 만족


print('\n독립성 : 독립변수의 값이 서로 관련 되지 않아야 한다')
# 듀빈-왓슨(Durbin-Watson) 검정으로 확인
# Durbin-Watson = 1.935 → 2에 근사하면 자기상관 없음
import statsmodels.api as sm
print('Durbin-Watson : ', sm.stats.stattools.durbin_watson(residual)) # 1.931498127082959 → 2에 근사하면 자기상관 없음

print('\n등분산성 : ')

print('\n다중공선성 : ')

