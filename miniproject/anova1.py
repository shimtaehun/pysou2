import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


data = pd.read_csv('miniproject/filtered_MarketingData.csv')

print('MntWines')
group0 = data[data['Kidhome'] == 0]['MntWines']
group1 = data[data['Kidhome'] == 1]['MntWines']
group2 = data[data['Kidhome'] == 2]['MntWines']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntWines   f-static : {f_static}, p-value : {p_value}')
print()

print('MntFruits')
group0 = data[data['Kidhome'] == 0]['MntFruits']
group1 = data[data['Kidhome'] == 1]['MntFruits']
group2 = data[data['Kidhome'] == 2]['MntFruits']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntFruits   f-static : {f_static}, p-value : {p_value}')
print()

group0 = data[data['Kidhome'] == 0]['MntMeatProducts']
group1 = data[data['Kidhome'] == 1]['MntMeatProducts']
group2 = data[data['Kidhome'] == 2]['MntMeatProducts']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntMeatProducts   f-static : {f_static}, p-value : {p_value}')
print()

group0 = data[data['Kidhome'] == 0]['MntFishProducts']
group1 = data[data['Kidhome'] == 1]['MntFishProducts']
group2 = data[data['Kidhome'] == 2]['MntFishProducts']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntFishProducts   f-static : {f_static}, p-value : {p_value}')
print()

group0 = data[data['Kidhome'] == 0]['MntSweetProducts']
group1 = data[data['Kidhome'] == 1]['MntSweetProducts']
group2 = data[data['Kidhome'] == 2]['MntSweetProducts']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntSweetProducts   f-static : {f_static}, p-value : {p_value}')
print()

group0 = data[data['Kidhome'] == 0]['MntGoldProds']
group1 = data[data['Kidhome'] == 1]['MntGoldProds']
group2 = data[data['Kidhome'] == 2]['MntGoldProds']
f_static , p_value = f_oneway(group0, group1, group2)
print(f' MntGoldProds   f-static : {f_static}, p-value : {p_value}')
print()

from scipy.stats import shapiro

stat, p = shapiro(data.MntWines)
print("p-value:", p)
if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")

stat, p = shapiro(data.MntFruits)
print("p-value:", p)
if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")

stat, p = shapiro(data.MntMeatProducts)
print("p-value:", p)
if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")

stat, p = shapiro(data.MntFishProducts)
print("p-value:", p)
if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")

stat, p = shapiro(data.MntSweetProducts)
print("p-value:", p)
if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")

stat, p = shapiro(data.MntGoldProds)
print("p-value:", p)
if p > 0.05:
    print("정규성을 만족함")
else:
    print("정규성을 만족하지 않음")


from scipy.stats import kruskal
group0 = data[data['Kidhome'] == 0]['MntWines']
group1 = data[data['Kidhome'] == 1]['MntWines']
group2 = data[data['Kidhome'] == 2]['MntWines']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)
if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntFruits']
group1 = data[data['Kidhome'] == 1]['MntFruits']
group2 = data[data['Kidhome'] == 2]['MntFruits']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)
if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntMeatProducts']
group1 = data[data['Kidhome'] == 1]['MntMeatProducts']
group2 = data[data['Kidhome'] == 2]['MntMeatProducts']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)
if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntFishProducts']
group1 = data[data['Kidhome'] == 1]['MntFishProducts']
group2 = data[data['Kidhome'] == 2]['MntFishProducts']
stat, p = kruskal(group0, group1, group2)

print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)
if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntSweetProducts']
group1 = data[data['Kidhome'] == 1]['MntSweetProducts']
group2 = data[data['Kidhome'] == 2]['MntSweetProducts']
stat, p = kruskal(group0, group1, group2)
print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)
if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

group0 = data[data['Kidhome'] == 0]['MntGoldProds']
group1 = data[data['Kidhome'] == 1]['MntGoldProds']
group2 = data[data['Kidhome'] == 2]['MntGoldProds']
stat, p = kruskal(group0, group1, group2)
print("Kruskal-Wallis H 통계량:", stat)
print("p-value:", p)
if p < 0.05:
    print("→ 유의미한 차이 있음 (귀무가설 기각)")
else:
    print("→ 유의미한 차이 없음 (귀무가설 채택)")

import scikit_posthocs as sp

dunn_result = sp.posthoc_dunn(data, val_col='MntWines', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)

plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntFruits', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()

dunn_result = sp.posthoc_dunn(data, val_col='MntMeatProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()




dunn_result = sp.posthoc_dunn(data, val_col='MntFishProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()



dunn_result = sp.posthoc_dunn(data, val_col='MntSweetProducts', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()




dunn_result = sp.posthoc_dunn(data, val_col='MntGoldProds', group_col='Kidhome', p_adjust='bonferroni')
print(dunn_result)
plt.figure(figsize=(8, 6))
sns.heatmap(dunn_result, annot=True, fmt=".2e", cmap="coolwarm", linewidths=0.5)
plt.title("Dunn's Test Pairwise p-values (Bonferroni corrected)")
plt.xlabel("Group")
plt.ylabel("Group")
plt.show()

features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
x = data[features]
y = data['Kidhome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.4f}\n")

# 4. 변수 중요도 확인
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n변수 중요도:")
print(feature_importances)

# 변수 중요도 시각화
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="Blues")
plt.title("변수 중요도 (RandomForest)")
plt.xlabel("중요도 점수")
plt.ylabel("종속변수")
plt.show()

# 혼동행렬 모델 성능 평가
print(classification_report(y_test, y_pred))

# 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0명','1명','2명'])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix 혼동행렬")
plt.xlabel('예측')
plt.ylabel('실제')
plt.show()