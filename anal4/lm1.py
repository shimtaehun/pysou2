# 최소 제곱해를 선형 행렬 방정식으로 구하기

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.5, 2.1])
# plt.scatter(x,y)
# plt.show()

A = np.vstack([x, np.ones(len(x))]).T
print(A)

import numpy.linalg as lin
# y = wx + b라는 일차 방정식의 w와 b의 값을 구하기
w, b = lin.lstsq(A, y, rcond=None)[0]   # 최소제곱법 연산
# 최소제곱법: 잔차 제곱의 총합이 최소가 되는 값을 얻을 수 있다.
print('w(weight, 기울기, slope): ', w)                                   
print('b(bias, 절편, 편향, intercept): ', b)
# 1차식의 완성본(y = 0.95999 * x + -0.98999)    
# 단순선형회기수식(모델)
plt.scatter(x,y)
plt.plot(x, w * x + b, label='실제값')
plt.legend()
plt.show()
# 수식으로 예측값 얻기
print(w * 1 + b)    # -0.0299(예측값)  -  0.2(실제값)   <== 잔차, 오차, 손실, 에러

