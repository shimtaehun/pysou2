# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
# kind quantity
# 1 64
# 2 72
# 3 68
# 4 77
# 2 56
# 1 NaN
# 3 95
# 4 78
# 2 55
# 1 91
# 2 63
# 3 49
# 4 70
# 1 80
# 2 90
# 1 33
# 1 44
# 3 55
# 4 66
# 2 77
# 귀무가설: 기름의 종류에 따라 빵에 흡수된 기름의 양의 평균에는 차이가 없다
# 대립가설: 기름의 종류에 따라 빵에 흡수된 기름의 양의 평균에는 차이가 있다.
import pandas as pd
import numpy as np
from scipy import stats
data = {'kind': [1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 3, 4, 2],
'quantity': [64, 72, 68, 77, 56, np.nan, 95, 78, 55, 91, 63, 49, 70, 80, 90, 33, 44, 55, 66, 77]}

df = pd.DataFrame(data)
quantity_mean = df['quantity'].mean()
df.fillna(quantity_mean,inplace=True)
print(df)
group1 = df[df['kind'] ==1]['quantity']
group2 = df[df['kind'] ==2]['quantity']
group3 = df[df['kind'] ==3]['quantity']
group4 = df[df['kind'] ==4]['quantity']
print(stats.f_oneway(group1, group2, group3, group4))
# pvalue: 0.8482436666841788 > 0.05 이므로 귀무가설 채택


