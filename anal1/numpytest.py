import numpy as np
# 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.

# 5행 4열 구조의 다차원 배열 객체를 생성
data = np.random.randn(5, 4)
print(data) 
# print(np.add(data))

# 각 행 단위의 합계
sums = np.sum(data, axis=1)
print(sums)

# 최댓값 구하기
maxs = np.max(data)
print(maxs)

