# 선형회귀 평가 지표 관련

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 공부 시간에 따른 시험 점수 데이터 생성: 표본 수 16
df = pd.DataFrame({'studytime': [3,4,5,8,10,5,8,6,3,6,10,9,7,0,1,2],
                'score':[76,74,74,89,66,75,84,82,73,81,95,88,83,40,70,69]})
print(df.head(3))

# dataset 분리: train / test data   - sort는 절대 XXXXXXXXXXX(왜곡된 자료로 분리하면 안돼!!!!!!!)
train, test = train_test_split(df, test_size=0.4, random_state=1)
print(len(train), len(test))
x_train = train[['studytime']]
y_train = train['score']
x_test = test[['studytime']]
y_test = test['score']
print('x_train', x_train)
print("y_train", y_train)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print()
model = LinearRegression()
model.fit(x_train, y_train)     # 모델 학습은 train data를 사용
y_pred = model.predict(x_test)  # 모델 평가용 예측은 test data를 사용
print('예측값: ', np.round(y_pred, 0))  # 예측값:  [85. 66. 80. 78. 85. 90. 90.]
print('실제값: ', y_test.values)        # 실제값:  [89. 40. 82. 74. 84. 95. 66.]

print('모델의 성능은? - r2_score, MSE가 일반적')
# 결정계수 수식으로 직접 작성 후 api 메소드와 비교
# 잔차 구하기
y_mean = np.mean(y_test)    # y의 평균
# 오차 제곱합: sum(y실제값 - y예측값)²
bunja = np.sum(np.square(y_test - y_pred))
# 편차 제곱합: sum(y관측값 - y평균값)²
bunmo = np.sum(np.square(y_test - y_mean))
# 실제값과 관측값은 같은 말이다.
r2 = 1 - (bunja / bunmo)  # 1 - (오차제곱합 / 편차제곱합)
print('계산에 결정계수: ', r2)

from sklearn.metrics import r2_score
print('api 제곱의 메소드 결과계수: ', r2_score(y_test, y_pred))

# R² 값은 분산을 기반으로 측정하는 도구인데 중심극한정리에 의해 표본 데이터가 많아지면 그 수치도 증가한다.
import seaborn as sns
import matplotlib.pyplot as plt

def linearFunc(df, test_size):
    train, test = train_test_split(df, train_size=test_size, shuffle=True, random_state=2)
    x_train = train[['studytime']]
    y_train = train['score']
    x_test = test[['studytime']]
    y_test = test['score']

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # R² 계산
    print('R²의 값: ', r2_score(y_test, y_pred))
    print('test data 비율: 전체 데이터 수의 {0}%'.format(test_size * 100))
    print('데이터 수 :{0}개'.format(x_train))
    # 시각화
    sns.scatterplot(x = df['studytime'], y = df['score'], color='green')
    sns.scatterplot(x = x_test['studytime'], y = y_test, color='red')
    sns.lineplot(x = x_test['studytime'], y = y_pred, color='blue')
    plt.show()

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # test 자료 수를 10%에서 50%로 늘려가며 R² 값 구하기
for i in test_sizes:
    linearFunc(df, i)



