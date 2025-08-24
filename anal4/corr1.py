# 공분산, 상관계수
# 공분산: 두 변수의 패턴을 확인하기 위해 공분산을 사용. 단위 크기에 영향을 미침
# 상관계수: 공분산을 표준화. -1 ~ 0 ~ 1, +-1에 근사하면 관계가 강함
import numpy as np
# 공분산: 패턴의 방향은 알겠으나 구체적인 크기를 표현은 곤란
print(np.cov(np.arange(1, 6), np.arange(2,7)))    # 1부터 5까지의 배열과 2부터 6까지의 배열의 공분산 행렬을 계산
print(np.cov(np.arange(10, 60, 10), np.arange(20,70,10)))   # 10부터 50까지 10씩 증가하는 배열과 20부터 70까지 10씩 증가하는 배열의 공분산 행렬을 계산
print(np.cov(np.arange(100, 600, 100), np.arange(200, 700, 100)))   # 100부터 500까지 100씩 증가하는 배열과 200부터 700까지 100씩 증가하는 배열의 공분산 행렬
print(np.cov(np.arange(1, 6), (3,3,3,3,3)))
print(np.cov(np.arange(1, 6), np.arange(6,1,-1)))
print('-'*60)
x = [8,3,6,6,9,4,3,9,3,4]
print('x의 평균: ', np.mean(x))
print('x의 분산: ', np.var(x))
y = [6,2,4,6,9,5,1,8,4,5]
print('y의 평균: ', np.mean(y))
print('y의 분산: ', np.var(y))  

import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.show()
# plt.close()
print('x, y 공분산: ', np.cov(x, y))    # 52.22
print('x, y 공분산: ', np.cov(x, y)[0, 1])
print()
print('x, y 상관계수: ', np.corrcoef(x, y))     # 0.8663
print('x, y 상관계수: ', np.corrcoef(x, y)[0, 1])

# 참고: 비선형인 경우는 일반적인 상관계수 방법을 사용하면 안됨
m = [-3, -2, -1, 0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]
plt.scatter(m, n)
plt.show()
plt.close()
print('m, n 상관계수: ', np.corrcoef(m, n)[0, 1])   # 무의미한 작업

