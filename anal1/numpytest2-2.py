import numpy as np

data = np.zeros((6, 4))
# 문 2-2) 6행 4열의 다차원 zero 행렬 객체를 생성
a = np.random.randint(20, 100, size = 24).reshape(6,4)
print(a)
for i in range(0, a.shape[1]):
    a[:, i] = a[:, 0] + 1
print(a)

