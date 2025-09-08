# k-평균 알고리즘은 주어진 데이터를
# k개의 클러스터로 묶는 알고리즘으로, 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작한다. 
# 이 알고리즘은 자율 학습의 일종으로, 레이블이 달려 있지 않은 입력 데이터에 레이블을 달아주는 역할을 수행한다.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # cluster 연습용 dataset
x, _= make_blobs(n_samples=150, n_features=2, \
                centers=3, cluster_std=0.5, shuffle=True, random_state=0)
print(x[:5], x.shape)

# plt.scatter(x[:, 0], x[:, 1], c='gray', marker='o', s = 50)
# plt.grid()
# plt.show()

from sklearn.cluster import KMeans
init_centroid = 'random'    # 초기 클러스터 중심을 임의로 선택
# init_centroid = 'k-means++'

kmodel = KMeans(n_clusters=3, init=init_centroid, random_state=0)
pred = kmodel.fit_predict(x)
# print('pred: ', pred)
# print(x[pred == 0])

print('centroid: ', kmodel.cluster_centers_)

plt.scatter(x[pred == 0, 0], x[pred == 0, 1], c='red', marker='o', s=50, label='cluster1')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], c='yellow', marker='s', s=50, label='cluster2')
plt.scatter(x[pred == 2, 0], x[pred == 2, 1], c='blue', marker='v', s=50, label='cluster3')
plt.scatter(x[pred == 0, 0], x[pred == 0, 1], c='red', marker='o', s=50, label='cluster1')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], c='yellow', marker='s', s=50, label='cluster2')
plt.scatter(kmodel.cluster_centers_[:,0], kmodel.cluster_centers_[:, 1], c='black', marker='+', s=80, label='center')

plt.legend()
plt.grid()
plt.show()


import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm

# 데이터 X와 X를 임의의 클러스터 개수로 계산한 k-means 결과인 y_km을 인자로 받아 각 클러스터에 속하는 데이터의 실루엣 계수값을 수평 막대 그래프로 그려주는 함수를 작성함.
# y_km의 고유값을 멤버로 하는 numpy 배열을 cluster_labels에 저장. y_km의 고유값 개수는 클러스터의 개수와 동일함.
# 가장 합리적인 클러스터 중심점 갯수 구하기
# 방법 1) elbow 기법 - 클러스터 간 SSE의 차이를 이용해 최적의 클러스터 수 반환
def elbowFunc(x):
    sse = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', random_state=0)
        km.fit(x)
        sse.append(km.inertia_)
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('count cluster')
    plt.ylabel('sse')
    plt.show()  # cluster는 3의 값을 추천한다.

elbowFunc(x)

def plotSilhouette(x, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장
    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기
        c_sil_value = sil_val[pred == c]
        c_sil_value.sort()
        y_ax_upper += len(c_sil_value)

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_sil_value)

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 개수')
    plt.show()
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
km = KMeans(n_clusters=3, random_state=0) 
y_km = km.fit_predict(X)

plotSilhouette(X, y_km)

# 방법 2)  