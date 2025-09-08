# iris dataset으로 지도학습(K-NN) / 비지도학습(KMeans)

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print(iris_dataset.keys())

print(iris_dataset['data'][:3])
print(iris_dataset['feature_names'])
print(iris_dataset['target'][:3])
print(iris_dataset['target_names'])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_dataset['data'], \
                                                    iris_dataset['target'], test_size=0.25, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

print('지도학습: K-NN ----------', )
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics

knnModel = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
knnModel.fit(train_x, train_y)      # feature, label(tag, target, class )

predict_label = knnModel.predict(test_x)
print('예측값: ', predict_label)
print('test acc: {:.3f}'.format(np.mean(predict_label == test_y)))
print('acc: ', metrics.accuracy_score(test_y, predict_label))

#새로운 데이터 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
print(knnModel.predict(new_input))
print(knnModel.predict_proba(new_input))
dist, index = knnModel.kneighbors(new_input)
print(dist, index)

print('\n비지도학습: KMeans(데이터에 정답(label)이 없는 경우) ---------------')
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
kmeansModel.fit(train_x)    # label X
# print(kmeansModel.labels_)
print('0 cluster:', train_y[kmeansModel.labels_ == 0])
print('1 cluster:', train_y[kmeansModel.labels_ == 1])
print('2 cluster:', train_y[kmeansModel.labels_ == 2])

# 이번엔 클러스터링에서 새로운 데이터 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
clu_pred = kmeansModel.predict(new_input)
print(clu_pred)     # [2]

