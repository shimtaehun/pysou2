from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
# - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
# - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
# 참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

data = {
    '지상파':[0.9,1.2,1.2,1.9,3.3,4.1,5.8,2.8,3.8,4.8,np.nan,0.9,3.0,2.2,2.0],
    '종편':[0.7,1.0,1.3,2.0,3.9,3.9,4.1,2.1,3.1,3.1,3.5,0.7,2.0,1.5,2.0],
    '운동':[4.2,3.8,3.5,4.0,2.5,2.0,1.3,2.4,1.3,35.0,4.0,4.2,1.8,3.5,3.5]
}
df = pd.DataFrame(data)
jisang_mean = df['지상파'].mean()
df.fillna(jisang_mean,inplace=True)
print(df)
x = df.지상파
y = df.운동
model = stats.linregress(x, y)
print(model)
plt.scatter(x, y)
input_ex = 10000
print(model.slope * input_ex + model.intercept)
plt.plot()
plt.show()
# LinregressResult(slope=np.float64(1.6677694999166923), intercept=np.float64(0.6184430442731452), 
# rvalue=np.float64(0.29321860062489036), pvalue=np.float64(0.2888491285513351), 
# stderr=np.float64(1.5081736936099939), intercept_stderr=np.float64(4.606536630914341))7.2895210439399145
print('점수 예측(iq가 80): ', model.slope * 80 + model.intercept)
print('점수 예측(iq가 120): ', model.slope * 120 + model.intercept)
