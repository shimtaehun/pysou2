# Broadcasting: 크기가 다른 배열 간의 연산시 배열의 구조 자종 변환
# 작은배열과 큰배열 연산 시 작은 배열은 큰 배열에 구조를 따름

import numpy as np

x = np.arange(1, 10).reshape(3, 3)
y = np.array([1, 0, 1])
print(x)
print(y)
print('=============================')
# 두 배열의 요소더하기
# 1) 새로운 배열을 이용
z = np.empty_like(x)
print(z)
for i in range(3):
    z[i] = x[i] + y
print(z)

# 2) tile을 이용
kbs = np.tile(y, (3, 1))
print('kbs : ', kbs)
z = x + kbs
print(z)
print('----------------')
# 3) Broadcasting 이용
# 1D + 1D(같은 길이), 1D + 1D(한쪽 길이1), 2D + 1D 가능
# 1D + 1D(길이 다르고 1도 아님) 불가능
kbs = x + y
print(kbs)

print()

a = np.array([0,1,2])
b = np.array([5,5,5])
print(a + b)
print(a + 5)

print('\n넘파이로 파일 i/o')
np.save('numpy4etc', x)     # 기본값은 binary 형식으로 저장
np.savetxt('numpy4etc.txt', x)  # txt 형식으로 저장

imsi = np.load('numpy4etc.npy')
print(imsi)

mydatas = np.loadtxt("numpy4etc2.txt", delimiter=',')
print(mydatas)