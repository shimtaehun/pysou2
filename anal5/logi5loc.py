# 분류모델 성능 평가 관련 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
print(x[:3])
print(y[:3])

import matplotlib.pyplot as plt
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

model = LogisticRegression().fit(x, y)
yhat = model.predict(x)
print('yhat: ', yhat[:3])

f_value = model.decision_function(x) # 결정함수(판별함수, 불확실성 추정함수), 판별 경계선 설정을 위한 샘플 자료 얻기
print('f_value: ', f_value[:10])

df = pd.DataFrame(np.vstack([f_value, yhat, y]).T, columns=["f", "yhat", "y"])
print(df.head(3))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, yhat))
acc = (44 + 44) / 100
recall = 44 / (44 + 4)
precion = 44 / (44 + 8)
specificity = 44 / (8 + 44)     # TN / FP + TN 
fallout = 8 / (8 + 44)          # 위양성율  FP / (FP + TN)
print('acc(정확도): ', acc)
print('recall(재현율): ', recall)
print('precion(정밀도): ', precion)
print('specificity(특이도): ', specificity)
print('fallout(위양성율)1: ', fallout)
print('fallout(위양성율)2: ', 1 - specificity)
#정리하면 TPR은 1에 근사하면 좋고, FPR은 0에 근사하면 좋다.
print()
from sklearn import metrics
ac_sco = metrics.accuracy_score(y, yhat)
print('acc(정확도): ', ac_sco)
cl_rep = metrics.classification_report(y, yhat)
print("cl_rep: \n", cl_rep)
print()
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print("fpr: ", fpr)
print("tpr: ", tpr)
print('분류임계결정값: ', thresholds)

# ROC 커브 시각화
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0, 1], [0, 1], 'k--', label='random classifier line(AUC 0.5)')
plt.plot([fallout], [recall], 'ro', ms=10)  # 위양성률과 재현률 값 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()
plt.close()

# AUC(Area Under The Curve) - ROC 커브의 면적
# 1에 가까울수록 좋은 분류모델로 평가된다.
print('AUC: ', metrics.auc(fpr, tpr))