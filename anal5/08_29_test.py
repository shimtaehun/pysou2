import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf


# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값0(착용X), 값1(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X

data = pd.read_csv('https://raw.githubusercontent.com/shimtaehun/miniproject/refs/heads/main/marketing_campaign%20(1).csv')
df = pd.DataFrame(data)
print(df.head(3))

# train, test = train_test_split(data, test_size=0.3, random_state=42)
# formula = '안경유무 ~ 게임 + TV시청'  
# logit_model = smf.logit(formula=formula, data=train).fit()
# print("logit: ", logit_model)