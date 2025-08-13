import numpy as np

a = np.random.randn(4, 5)
print(a)


avr = np.mean(a)
sums = np.sum(a)
arr = np.std(a)
var = np.var(a)
maxs = np.max(a)
mins = np.min(a)
per1 = np.percentile(a, 25)
per2 = np.percentile(a, 50)
per3 = np.percentile(a, 75)
per4 = np.cumsum(a)

print('평균: ', avr)
print('합계:', sums)
print('표준편차:', arr)
print('분산:', var)
print('최댓값:', maxs)
print('최솟값:', mins)
print('1사분위 수:', per1)
print('2사분위 수:', per2)
print('3사분위 수:', per3)
print('요소값 누적합:', per4)

