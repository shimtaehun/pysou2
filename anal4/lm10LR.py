#  sklearn 모듈의 LinearRegression 클래스 사용
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 표준화와 정규화
import matplotlib.pyplot as plt
sample_size = 100
np.random.seed(1)

# 1) 편차가 없는 데이터 생성
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 60
print(x[:5])
print(y[:5])
print('상관계수: ', np.corrcoef(x,y))

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled', x_scaled[:5])
# plt.scatter(x_scaled, y)
# plt.show()
# 위 작업이 정규화 작업

model = LinearRegression().fit(x_scaled, y)
print(model)
print('계수(slope): ', model.coef_)     # 회귀계수 (각 독립변수가 종속변수에 미치는 영향)
print('절편(intercept): ', model.intercept_)    
print('결정계수(R²): ', model.score(x_scaled, y))   # 설명력: 훈련 데이터 기준
# y = wx + b <== 2696.55038642 * x + -1381.6493752394929
y_pred = model.predict(x_scaled)
print('예측값(ŷ): ', y_pred[:5])   # [ 977.62741972 -366.1674945  -315.93693523 -643.3349412   521.54054659]
print('실제값(ŷ): ', y[:5])    # [ 970.13593255 -354.80877114 -312.86813494 -637.84538806  508.29545914]
# model.summary()   지원 X
print('-'*50)
# 선형회귀는 MAE는 실제 값과 예측값의 절대 오차 평균이며, MSE는 오차 제곱의 평균, RMSE는 MSE에 제곱근을 씌운 값으로 큰 오차에 더 민감합니다. 
# R²는 실제 값의 변동 중에서 모델이 설명하는 변동의 비율을 나타냅니다. 
# 모델 성능 파악용 함수 작성
def RegScoreFunc(y_true, y_pred):
    print('R²_score(결정계수):{}'.format(r2_score(y_true, y_pred)))
    print('설명분산점수:{}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_error(평균제곱오차):{}'.format(mean_squared_error(y_true, y_pred)))

RegScoreFunc(y, y_pred)
# R²_score(결정계수):0.9996956382642653
# 설명분산점수:0.9996956382642653
# mean_squared_error(평균제곱오차):86.14795101998757

print('-'*100)

# 1) 편차가 있는 데이터 생성
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관계수: ', np.corrcoef(x,y))   # 0.00401167

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled', x_scaled[:5])
plt.scatter(x_scaled, y)
plt.show()
# 위 작업이 정규화 작업

model = LinearRegression().fit(x_scaled, y)
print(model)
y_pred = model.predict(x_scaled)
print('예측값(ŷ): ', y_pred[:5])    # 예측값(ŷ):  [-10.75792685  -8.15919008 -11.10041394  -5.7599096  -12.73331002]
print('실제값(ŷ): ', y[:5])         # 실제값(ŷ):  [1020.86531436 -710.85829436 -431.95511059 -381.64245767 -179.50741077]

RegScoreFunc(y, y_pred)
# R²_score(결정계수):1.6093526521765433e-05
# 설명분산점수:1.6093526521765433e-05
# mean_squared_error(평균제곱오차):282457.9703485092