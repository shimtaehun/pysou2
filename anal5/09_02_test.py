import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 1. 데이터 불러오기
wine_df = pd.read_csv('anal5\winequality-red.csv')
print(wine_df.head(5))

# 2. 데이터 탐색 (EDA)
# 품질(quality) 변수의 분포 확인
plt.figure(figsize=(8, 5))
plt.title('레드 와인 품질 분포')
plt.xlabel('품질 점수')
plt.ylabel('개수')
plt.show()

# 3. 데이터 전처리
# 특성(X)과 타겟(y) 변수 분리
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']

# 학습 데이터와 테스트 데이터 분리 (80:20 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n학습 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")


rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# 5. 모델 평가
# 테스트 데이터로 예측
y_pred = rf_clf.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- 모델 평가 ---")
print(f"정확도(Accuracy): {accuracy:.4f}")

# 분류 리포트 출력 (정밀도, 재현율, F1-점수)
print("\n--- 분류 리포트 ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 혼동 행렬(Confusion Matrix) 시각화
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('혼동 행렬 (Confusion Matrix)')
plt.xlabel('예측된 값')
plt.ylabel('실제 값')
plt.show()

# 6. 특성 중요도 확인
feature_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_sorted, y=feature_importances_sorted.index)
plt.title('특성 중요도 (Feature Importances)')
plt.xlabel('중요도')
plt.ylabel('특성')
plt.show()

print("\n--- 특성 중요도 ---")
print(feature_importances_sorted)