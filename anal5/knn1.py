# 필요한 라이브러리들을 가져옵니다.
from sklearn.datasets import load_breast_cancer  # scikit-learn에서 유방암 데이터셋을 불러오기 위한 함수
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 테스트용으로 나누기 위한 함수
from sklearn.neighbors import KNeighborsClassifier  # KNN 분류 모델
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리

# 유방암 데이터셋을 로드합니다.
cancer = load_breast_cancer()

# 데이터를 훈련 세트와 테스트 세트로 나눕니다.
# cancer.data는 문제지(특성), cancer.target은 정답(레이블)입니다.
# stratify=cancer.target은 훈련/테스트 데이터의 클래스 비율을 원본 데이터와 동일하게 유지합니다.
# random_state=66은 재현성을 위해 난수 시드를 고정합니다.
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# 훈련 세트와 테스트 세트의 정확도를 저장할 리스트를 초기화합니다.
train_accuracy = []
test_accuracy = []

# 이웃의 수(k)를 1부터 10까지 2씩 증가시키며 테스트하기 위한 범위를 설정합니다. (1, 3, 5, 7, 9)
neighbors_set = range(1, 11, 2)

# 설정된 이웃의 수(k) 범위에 대해 반복합니다.
for n_neighbor in neighbors_set:
    # KNN 분류기 모델을 생성합니다.
    # n_neighbors는 이웃의 수(k)를 의미합니다.
    # p=1은 맨해튼 거리를 사용하도록 설정합니다 (p=2는 유클리드 거리).
    # metric='minkowski'는 거리 측정 방법을 민코프스키 거리로 지정합니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbor, p=1, metric='minkowski')

    # 훈련 데이터를 사용하여 모델을 학습시킵니다.
    clf.fit(x_train, y_train)
    
    # 훈련 세트에 대한 모델의 정확도를 계산하고 리스트에 추가합니다.
    train_accuracy.append(clf.score(x_train, y_train))
    
    # 테스트 세트에 대한 모델의 정확도를 계산하고 리스트에 추가합니다.
    test_accuracy.append(clf.score(x_test, y_test))

# numpy 라이브러리를 가져옵니다. 숫자 계산에 유용합니다.
import numpy as np

# 각 k값에 대한 훈련 정확도의 평균을 출력합니다.
print('train 분류 정확도 평균: ', np.mean(train_accuracy))
# 각 k값에 대한 테스트 정확도의 평균을 출력합니다.
print('test 분류 정확도 평균: ', np.mean(test_accuracy))

# k값의 변화에 따른 훈련 정확도를 그래프로 그립니다.
plt.plot(neighbors_set, train_accuracy, label='train_accuracy')
# k값의 변화에 따른 테스트 정확도를 그래프로 그립니다.
plt.plot(neighbors_set, test_accuracy, label='test_accuracy')

# 그래프의 y축에 'acc' (accuracy) 라벨을 추가합니다.
plt.ylabel('acc')
# 그래프의 x축에 'k' (number of neighbors) 라벨을 추가합니다.
plt.xlabel('k')
# 그래프에 범례를 표시합니다.
plt.legend()
# 그래프를 화면에 보여줍니다.
plt.show()
