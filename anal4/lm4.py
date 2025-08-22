# 방법4: linregress     model 0
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 값 예측
score_iq = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())

x = score_iq.iq
y = score_iq.score

# 상관 계수 확인
print(np.corrcoef(x, y)[0,1])   # 0.88222
print(score_iq.corr())
plt.scatter(x, y)
# plt.show()
# plt.close()

model = stats.linregress(x, y)
print(model)    # (slope: (0.6514309527270075), intercept: (-2.8564471221974657), rvalue: (0.8822203446134699), pvalue: (2.8476895206683644e-50), stderr: (0.028577934409305443), intercept_stderr: (3.546211918048538))
print('기울기: ', model.slope)          # 0.6514309527270075
print('절편: ', model.intercept)        # -2.8564471221974657
print('R²- 결정계수: ', model.rvalue)    # 0.8822203446134699 : 독립변수가 종속변수를 88% 정도 설명하고 있다. 모델의 성능이 좋다.
print('p-value: ', model.pvalue)        # 2.8476895206683644e-50   < 0.05 이므로 현재 모델은 유의하다. (독립변수와 종속변수는 인과관계가 있다.)
print('표준오차: ', model.stderr)        #0.028577934409305443
# y = wx + b => 0.651430952727 * x + -2.8564471221
plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept)  # 관통을 하고 있는지 확인하기 위해서 사용한 코드
plt.show()
plt.close()

# 점수 예측
print('점수 예측(iq가 80): ', model.slope * 80 + model.intercept)
print('점수 예측(iq가 120): ', model.slope * 120 + model.intercept)
#predict X 
print('점수 예측(predict): ', \
np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))
print()
newdf = pd.DataFrame({'iq':[55, 66, 77, 88, 150]})
print('점수 예측(predict): ', \
np.polyval([model.slope, model.intercept], newdf))