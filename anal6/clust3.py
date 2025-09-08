# iris dataset을 이용한 계층적 군집분석
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns= iris.feature_names)
print(iris_df.head(2))
print(iris_df.loc[0:2, ['sepal length (cm)', 'sepal width (cm)']])

from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']],
                metric='euclidean')
print('dist_vec: ', dist_vec)
print()
row_dist = pd.DataFrame(squareform(dist_vec))
print("row_dist: ", row_dist)

from scipy.cluster.hierarchy import linkage, dendrogram
row_clusters = linkage(dist_vec, method='complete') # ward, average ...
print("row_clusters: \n", row_clusters)
df = pd.DataFrame(row_clusters, columns=['id1', 'id2', '거리', '멤버수'])
print(df)

row_dend = dendrogram(row_clusters)
plt.tight_layout()
plt.ylabel('dist')
plt.show()

print()
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
X = iris_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']]
labels = ac.fit_predict(X)
print('클러스터 분류 결과: ', labels)

plt.hist(labels)
plt.grid()
plt.show()
plt.close()