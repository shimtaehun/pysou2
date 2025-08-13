# 배열 연산
import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.arange(5, 9).reshape(2,2)
y = y.astype(np.float64)
print(x, x.astype, x.dtype)
print(y, y.astype, y.dtype)

# 요소별 합
print(x + y)    # python의 산술 연산자
print(np.add(x, y)) # numpy 
# np.subtaract, np.multiply, np.divide
import time
big_arr = np.random.rand(1000000)
start = time.time()
sum(big_arr) # python 함수
end = time.time()
print(f"sum():{end - start:.6f}sec")
start = time.time()
np.sum(big_arr) # numpy 함수
end = time.time()
print(f"np.sum():{end - start:.6f}sec")

# 요소별 곱
print(x)    # 1, 2, 3, 4
print(y)    # 5, 6, 7, 8
print(x * y)
print(np.multiply(x, y))

print(x.dot(y))# 내적 연산
print()
v = np.array([9, 10])
w = np.array([11, 12])
print(v * w)
print()
# 내적으로 구하기
print(v.dot(w))
print(np.dot(v, w))
print(np.dot(x, v))
print()
print(np.dot(x, y))

print('유용 함수 ---------------')
print(x)
print(np.sum(x))
print(np.sum(x, axis = 0))      # 열 단위 연산
print(np.sum(x, axis = 1))      # 행 단위 연산

print(np.min(x), '', np.max(x))     # 최소와 최대를 구함
print(np.argmin(x), '', np.argmax(x))   # 최소와 최대의 인덱스를 알려줌
print(np.cumsum(x))
print(np.cumprod(x))
print()

names = np.array(['tom', 'james', 'oscar','tom','oscar'])
names2 = np.array(['tom', 'page', 'john'])
print(np.unique(names))
print(np.intersect1d(names, names2))    # 교집합
print(np.intersect1d(names, names2, assume_unique=True))
print(np.union1d(names, names2)) 

print('\n전치(Transpose)')
print(x)
print(x.T)
arr = np.arange(1, 16).reshape(3,5)
print(arr)
print(arr.T)
print(np.dot(arr.T, arr))

print(arr. flatten())
print(arr.ravel())