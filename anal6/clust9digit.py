# 숫자 이미지 데이터에 K-평균 알고리즘 사용하기
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from scipy.stats import mode

# Seaborn 기본 스타일 설정
sns.set()

# 1. 데이터 로드
print("---" + " 1. 데이터 로드 " + "---")
digits = load_digits()  # 64개의 특징(feature)을 가진 1797개의 표본으로 구성된 숫자 데이터
print("데이터 형태:", digits.data.shape)  # (1797, 64) -> 8*8 이미지 픽셀
print("\n" + "-"*40 + "\n")


# 2. K-평균 클러스터링 수행
print("---" + " 2. K-평균 클러스터링 수행 " + "---")
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print("군집 중심 형태:", kmeans.cluster_centers_.shape)  # (10, 64) -> 64차원의 군집 10개
print("\n" + "-"*40 + "\n")


# 3. 군집 중심 시각화
# 결과를 통해 KMeans가 레이블 없이도 1과 8을 제외하면
# 인식 가능한 숫자를 중심으로 갖는 군집을 구할 수 있다는 사실을 알 수 있다.
print("---" + " 3. 군집 중심 시각화 " + "---")
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)

for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest')

plt.show()
print("\n" + "-"*40 + "\n")


# 4. 예측된 클러스터 레이블을 실제 레이블과 매칭
# k평균은 군집의 정체에 대해 모르기 때문에 0-9까지 레이블은 바뀔 수 있다.
# 이 문제는 각 학습된 군집 레이블을 그 군집 내에서 발견된 실제 레이블과 매칭해 보면 해결할 수 있다.
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]


# 5. 정확도 확인
print("---" + " 5. K-Means 기본 정확도 확인 " + "---")
print("정확도:", accuracy_score(digits.target, labels))
print("\n" + "-"*40 + "\n")


# 6. 오차 행렬(Confusion Matrix) 시각화
# 오차의 주요 지점은 1과 8에 있다.
print("---" + " 6. 오차 행렬 시각화 " + "---")
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()
print("\n" + "-"*40 + "\n")


# 7. t-SNE를 이용한 차원 축소 후 클러스터링
# 참고로 t분포 확률 알고리즘(t-SNE)을 사용하면 분류 정확도가 높아진다.
print("---" + " 7. t-SNE 적용 후 정확도 재확인 " + "---")
print("t-SNE 적용 중... (시간이 약간 걸릴 수 있습니다)")
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

# t-SNE로 변환된 데이터에 K-평균 재적용
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# 레이블 매칭
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# 정확도 재계산
print("t-SNE + K-Means 정확도:", accuracy_score(digits.target, labels))
print("\n" + "-"*40 + "\n")
