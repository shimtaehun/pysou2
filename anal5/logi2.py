# 날씨 예보(강우 여부)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print(data.head(3), data.shape)     # (366, 12)
data2 = pd.DataFrame()
data2 = data.drop(['Date', 'RainToday'], axis = 1)
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes': 1, 'No': 0})
print(data2.head(3), data2.shape)     # (366, 10)
print(data2.RainTomorrow.unique())

# 학습데이터와 검정데이터로 분리
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(train.shape, test.shape)  # (256, 10) (110, 10)
print(data2.columns)
col_select = "+".join(train.columns.difference(['RainTomorrow']))
print(col_select)
my_formula = 'RainTomorrow ~ ' + col_select

# model = smf.glm(formula = my_formula, data = train, family = sm.families.Binomial()).fit()
model = smf.logit(formula = my_formula, data = train).fit()

print(model.summary())
# print(model.params)
print('예측값: ', np.rint(model.predict(test)[:5].values))
print('실제값: ', test['RainTomorrow'][:5].values)

# 분류 정확도 
conf_tab = model.pred_table()
print('conf_tab: \n', conf_tab)     # pred_table 지원 안함
print('분류 정확도: ', (conf_tab[0][0] + conf_tab[1][1])/ len(train))

from sklearn.metrics import accuracy_score
pred = model.predict(test)  # 모델 만들기
print('분류 정확도: ', accuracy_score(test['RainTomorrow'], np.around(pred)))   # 0.87272