# matplotlib는 플로팅 모듈. 다양한 그래프 지원 함수 지원
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

"""
x= ['서울', '인천', '수원']
y = [5, 3, 7]
plt.xlim([-1, 3])
plt.ylim([0, 10])
plt.plot(x, y)
plt.yticks(list(range(0, 10, 3)))
plt.show()
# jupyter notebook에서는 '%matplotlib inline' 하면 show() 없어도 됨

data = np.arange(1, 11, 2)
print(data)     # [1 3 5  7 9] - 구간 4
plt.plot(data)
x = [0, 1, 2, 3, 4]
for a, b in zip(x, data):
    plt.text(a, b, str(b))
plt.show()

plt.plot(data)
plt.plot(data, data,'r')
for a, b in zip(data, data):
    plt.text(a, b, str(b))
plt.show()
"""


# sin 곡선
x = np.arange(10)
y = np.sin(x)
print(x, y)
# plt.plot(x, y)
# plt.plot(x, y, 'bo')    # 스타일 지정
# plt.plot(x, y, 'r+')
plt.plot(x, y, 'go--', linewidth=2, markersize=12)
# - (solid line), --(dashed line) ...
# color = 'b', c='b', lw=2, marker = 'o', ms = 12 ...
plt.show()

# 홀드 명령 : 하나의 영역에 두 개 이상의 그래프 표시
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y_sin, 'r')
plt.plot(x, y_cos)
plt.xlabel('x축')
plt.ylabel('y축')
plt.title('제목')

plt.legend(['sine', 'cosine'])
plt.show()
# subplot : Fifure를 여러 개 선언
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('사인')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('코사인')
plt.show()

print()
irum = ['a', 'b', 'c', 'd', 'e']
kor= [80, 50, 70, 70, 90]
eng = [60, 70, 80, 70, 60]
plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'gs-')
plt.ylim([0, 100])
plt.legend(['국어', '영어'], loc='best')     # 1, 2, 3, 4
plt.grid(True)
fig = plt.gcf()
plt.show()
fig.savefig('result.png')

from matplotlib.pyplot import imread
img = imread('result.png')
plt.imshow(img)
plt.show()