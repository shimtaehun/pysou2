# 클러스터링 기법 중 계층적 군집화 이해

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

np.random.seed(123)
var = ['x', 'y']
labels = ['점0', '점1', '점2', '점3', '점4']
x = np.random.random_sample([5, 2]) * 10
df = pd.DataFrame(x, columns=var, index=labels)
print(df)

# plt.scatter(x[:, 0], x[:, 1], c='blue', marker='o', s=50)
# plt.grid(True)
# plt.show()

from scipy.spatial.distance import pdist, squareform
# pdist: 배열에 있는 값을 이용해 각 요소들의 거리 계산
# squareform: 거리벡터를 사각형 형식으로 변환
dist_vec = pdist(df, metric='euclidean')
print("dist_vec: \n", dist_vec)

row_dist = pd.DataFrame(squareform(dist_vec),columns=labels, index=labels)
print(row_dist)

# 응집형: 자료 하나하나를 군집으로 보고 가까운 군집끼리 연결하는 방법. (상향식)
# 분리형: 전체 자료를 하나의 군집으로 보고 분리해 나가는 방법. 하향식

# linkage: 응집형 계층적 군집을 수행 (linkage)
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(dist_vec, method='ward')

df = pd.DataFrame(row_clusters, columns=['클러스트id_1', '클러스터id_2', '거리', '클러스터 멤버수'])
print(df)

# linkage의 결과로  덴드로그램 작성
from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('유클리드 거리')
plt.show()
plt.close()