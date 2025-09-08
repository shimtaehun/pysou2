# 앙상블 기법 중 부스팅: 하나의 모델을 만들어 그 결과로 다른 모델을 만들어 나가는데
# 샘플 데이터의 일부를 갱신하며 순차적으로 좀 더 강건한 모델을 생성하는 방법
# 가중치를 활용하여 약분류기를 강분류기로 만드는 방법

# brest_cancer dataset으로 분류 모델 
# pip install xgboost, lightgbm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from lightgbm import LGBMClassifier     # xgboost보다 성능 우수하나 데이터 양이 적으면 과적합 발생
import lightgbm as lgb

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=12, stratify=y
)

# 모델 학습 2개
xgb_clf = xgb.XGBClassifier(
    booster = 'gbtree',      # gbtree:DecisionTree, gblinear:DecisionTree를 사용하지 않는다.
    max_depth = 6,
    n_estimators = 500,
    eval_metric = 'logloss',        # error, rmse, ...
    random_state=42
)

lgb_clf = LGBMClassifier(n_estimators = 500, random_state=42, verbose=-1)
xgb_clf.fit(x_train, y_train)
lgb_clf.fit(x_train, y_train)

# 예측 / 평가
pred_xgb = xgb_clf.predict(x_test)
pred_lgb = lgb_clf.predict(x_test)
print(f'XGBoost acc: {accuracy_score(y_test, pred_xgb):.4f}')
print(f'LightGBM acc: {accuracy_score(y_test, pred_lgb):.4f}')
