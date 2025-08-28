# Logistic Regression
# 독립(feature, x):연속형, 종속(label, y):범주형
# 이항 분류(다항도 가능)
# 출력된 연속형(확률)자료를 logit 변환해, 최종적으로 sigmoid function에 의해 0 ~ 1 사이의 실수값이 나오는데 
# 0.5를 기준으로 0, 1로 분류한다.

# sigmoid function 살짝 맛보기
import math
def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))   # 로지스틱의 수식 f(x) = 1 / 1 + e^-x 를 나타낸것이다.z
# print(sigmoidFunc(3))
# print(sigmoidFunc(1))
# print(sigmoidFunc(-123))
# print(sigmoidFunc(0.123))

# mtcar dataset 사용
import statsmodels.api as sm    # 통계 분석을 위한 statsmodels 라이브러리에 있는 sm을 가져옴
mtcardata = sm.datasets.get_rdataset('mtcars')  # statsmodels에서 제공하는 'mtcars'데이터셋을 불러온다.
print(mtcardata.keys()) # dict_keys(['data', '__doc__', 'package', 'title', 'from_cache', 'raw_data']) 이런식으로 어떤 키가 있는지 표시해준다.
mtcars = mtcardata.data # 데이터셋에서 실제 데이터프레임 부분만 추출하여 mtcars 변수에 저장한다.
print(mtcars.head(2))
# mpg, hp가 am(자동, 수동)에 영향을 준다.
mtcar = mtcars.loc[:, ['mpg', 'hp', 'am']]  # 분석에 사용할 mpg, hp, am 세 개의 열만 선택하여 mtcar라는 데이터프레임을 만든다.
print(mtcar.head(2))
print(mtcar['am'].unique())     # am열에 어떤 고유한 값들이 있는지 확인한다. [1 0] (1은 수동 0은 자동)

# 연비와 마력수에 따른 변속기 분류 모델 작성(수동, 자동)
# 모델 작성 방법1: logit()
import statsmodels.formula.api as smf
formula = 'am ~ hp + mpg'   # 종속변수(am)와 독립변수(hp, mpg)의 관계를 수식으로 정의합니다. 'am ~ hp + mpg'는 am을 hp와 mpg로 예측하겠다는 의미이다.
model1 = smf.logit(formula=formula, data=mtcar).fit()   # logit() 함수를 사용하여 로지스틱 회귀 모델을 만들고, fit() 함수로 mtcar 데이터를 학습시킨다.
print(model1.summary()) # Logit Regression 결과표

# 예측값 / 실제값 출력
import numpy as np
import pandas as pd
# print('예측값: ', model1.predict())
pred = model1.predict(mtcar[:10])   # 학습된 모델을 사용하여 mtcar 데이터의 첫 10개 행에 대한 예측을 수행한다.
print('예측값: ', pred.values)  # 예측된 확률값을 출력한다.
print('예측값: ', np.around(pred.values))   # np.around는 값을 반올림 해줘서 0.5를 기준으로 0과 1로 변환하여 보여준다.
print('실제값: ', mtcar['am'][:10].values)  # 실제 변속기 값(am)의 첫 10개를 출력하여 예측값과 비교한다.
print('-'*100)
# 분류 모델의 정확도(accuracy) 확인
conf_tab = model1.pred_table()  # pred_table() 함수는 모델의 예측 결과를 바탕으로 혼동 행렬(Confusion Matrix)을 생성합니다.
print('confusion matrix: \n',conf_tab)  
# [[16.  3.]    16: 실제 0을 0으로 맞게 예측함(16개) 3: 실제 0을 1로 틀리게 예측함(3개)
# [ 3. 10.]]    3: 실제 1을 0으로 틀리게 예측(3개) 10: 실제 1을 1로 맞게 예측(10개)
# 혼동 행렬 해석:
# [[TP, FN],   - TN (True Negative): 실제 0을 0으로 맞게 예측 (16개)
#  [FP, TN]]   - FP (False Positive): 실제 0을 1로 틀리게 예측 (3개)
#              - FN (False Negative): 실제 1을 0으로 틀리게 예측 (3개)
#              - TP (True Positive): 실제 1을 1로 맞게 예측 (10개)
print('분류 정확도: ', (16 + 10) / len(mtcar))  # 계산: (올바르게 예측한 개수) / (전체 데이터 개)
print('분류 정확도: ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))  # 분류 정확도:  0.8125는 81프로의 확률로 맞춘다는 의미이다.
# 혼동 행렬 변수(conf_tab)를 인덱싱하여 정확도를 계산하고 출력한다. (TN + TP) / 전체 데이터              Logit Regression Results

from sklearn.metrics import accuracy_score
pred2 = model1.predict(mtcar)
print('분류 정확도: ', accuracy_score(mtcar['am'], np.around(pred2)))    # mtcar = 실제값   pred2 = 예측값

print('-'*100)
# 모델 작성 방법2: glm()
model2 = smf.glm(formula=formula, data = mtcar, family=sm.families.Binomial()).fit()    # Binomial() 이항분포
print('model2: ', model2)
print()
print(model2.summary())
glm_pred = model2.predict(mtcar[:10])
print('glm 예측값', np.around(glm_pred.values))
print('glm 실제값', mtcar['am'][:10].values)

glm_pred2 = model2.predict(mtcar)
print('glm 분류 정확도: ', accuracy_score(mtcar['am'], np.around(glm_pred2)))  

print('\n새로운 값(hp, mpg)으로 변속기(am) 분류 예측')
newdf = mtcar.iloc[:2].copy()
# print(newdf)
newdf['mpg'] = [10, 30]
newdf['hp'] = [120, 90]
print(newdf)
new_pred = model2.predict(newdf)
print('새로운 값(hp, mpg)에 대한 변속기는: ', np.around(new_pred.values))
print()
import pandas as pd
newdf2 = pd.DataFrame({'mpg': [10, 30, 50, 5], 'hp':[80, 110, 130, 50]})
new_pred2 = model2.predict(newdf2)
print('new_pred2', new_pred2.values)
print('new_pred2', np.around(new_pred2.values))
print('new_pred2', np.rint(new_pred2.values))

'''
==============================================================================
Dep. Variable:                     am   No. Observations:                   32
Model:                          Logit   Df Residuals:                       29
Method:                           MLE   Df Model:                            2
Date:                Wed, 27 Aug 2025   Pseudo R-squ.:                  0.5551
Time:                        17:09:59   Log-Likelihood:                -9.6163
converged:                       True   LL-Null:                       -21.615
Covariance Type:            nonrobust   LLR p-value:                 6.153e-06
==============================================================================
                coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
hp             0.0550      0.027      2.045      0.041       0.002       0.108
mpg            1.2596      0.567      2.220      0.026       0.147       2.372
==============================================================================

p > |z|에 hp와 mpg의 값이 0.05보다 크냐 작냐가 중요하다
'''

