import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 학생 10명의 시험점수로 KMeans 수행 K = 3
students = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
# 시험 점수
scores = np.array([76, 95, 65, 85, 60, 92, 55, 88, 83, 72]).reshape(-1,1)
print('점수: ', scores.ravel())

Kmeans = KMeans(n_clusters=3, random_state=0)
Kmeans_clust = Kmeans.fit_predict(scores)

df = pd.DataFrame({
    'Student':students,
    'Score':scores.ravel(),
    'Cluster':Kmeans_clust
})
print('군집 결과: \n', df)

print('군집별 평균 점수')
grouped = df.groupby('Cluster')['Score'].mean()
print(grouped)

# 시각화 
x_positions = np.arange(len(students))
y_scores = scores.ravel()
colors = {0:'red', 1:'blue', 2:'black'}

plt.figure(figsize=(10, 6))
for i, (x, y, cluster) in enumerate(zip(x_positions, y_scores, Kmeans_clust)):
    plt.scatter(x, y, color=colors[cluster], s=100)
    plt.text(x, y + 1.5, students[i], fontsize=10, ha='center')

# 중심점 표시
centers = Kmeans.cluster_centers_
for center in centers:
    plt.scatter(len(students) // 2, center[0], marker='X', c='gold', s=200)

plt.xticks(x_positions, students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.title('KMeans Clustering of Student Scores')
plt.grid()
plt.show()
plt.close()
