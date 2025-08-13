import numpy as np

# 1) 6행 6열의 다차원 행렬 만들기
data = np.zeros((6, 6))
print(data)

# 조건1) 36개의 셀에 1~36 까지 정수 채우기
a = np.arange(1, 37).reshape(6,6)
print(a)

# 조건2) 2번쨰 행 전체 원소 출력하기
print(a[1])

# 조건3) 5번째 열 전체 원소 출력하기
print(a[:,4])

# 조건4) 15~29 까지의 결과 출력하기

