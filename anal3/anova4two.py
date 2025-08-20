# 이원분산 분석: 두 개의 요인에 대한 집단(독립변수) 각각이 종속변수의 평균에 영향을 주는지 검정
# 가설이 주효과 2개, 교호작용 1개가 나옴
# 교호작용(interaction term): 한 쪽 요인이 취하는 수준에 따라 다른 쪽 요인이 
# 영향을 받는 요인의 조합효과를 말하는 것으로 상승과 상쇄효과가 있다.
# 예) 초밥과 간장, 감자튀김과 간장, 초밥과 케찹 ...

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 실습 1) 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향을 주는가?
# 주효과 가설
# 귀무: 태아 수와 태아의 머리둘레 평균은 차이가 없다.
# 대립: 태아 수와 태아의 머리둘레 평균은 차이가 있다.
# 귀무: 태아 수와 관측자 수의 머리둘레 평균은 차이가 없다.
# 대립: 태아 수와 관측자 수의 머리둘레 평균은 차이가 있다.
# 교호작용 가설
# 귀무: 교호작용이 없다. (태아수와 관측자수는 관련이 없다.)
# 대립: 교호작용이 있다. (태아수와 관측자수는 관련이 있다.)


url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt'
data = pd.read_csv(url)
print(data.head(3), data.shape)    # (36, 3)
print(data['태아수'].unique())      # [1 2 3]
print(data['관측자수'].unique())    # [1 2 3 4]

# reg = ols("머리둘레 ~ C(태아수) + C(관측자수)", data=data).fit()    # 교호작용 확인 X
# reg = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)", data=data).fit()    # 교호작용 확인 X
reg = ols("머리둘레 ~ C(태아수) * C(관측자수)", data=data).fit()    # 교호작용 확인 X
result = anova_lm(reg, type=2)
print(result)
#                   df      sum_sq     mean_sq            F        PR(>F)
# C(태아수)           2.0  324.008889  162.004444  2113.101449  1.051039e-27 < 0.05 귀무 기각
# C(관측자수)          3.0    1.198611    0.399537     5.211353  6.497055e-03 > 0.05 귀무 채택
# C(태아수):C(관측자수)   6.0    0.562222    0.093704     1.222222  3.295509e-01 > 0.05 귀무 채택
# 태아 수는 머리둘레에 강력한 영향을 미침. 관측자 수는 유의한 영향을 미침.
# 하지만 태아수와 관측자 수의 상호작용은 유의하지 않다.

print()
# 실습2: poison 종류와 treat(응급처치)가 독퍼짐 시간의 평균에 영향을 주는가?
# 주효과 가설
# 귀무: poison 종류와 독퍼짐 시간의 평균에 차이가 없다.
# 대립: poison 종류와 독퍼짐 시간의 평균에 차이가 있다.
# 귀무: treat(응급처치) 종류와 독퍼짐 시간의 평균에 차이가 없다.
# 대립: treat(응급처치) 종류와 독퍼짐 시간의 평균에 차이가 있다.

# 교호작용 가설
# 귀무: 교호작용이 없다. (poison 종류와 treat(응급처치) 방법은 관련이 없다.)
# 대립: 교호작용이 있다. (poison 종류와 treat(응급처치) 방법은 관련이 있다.)

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv', index_col=0)
print(data2.head(3), data2.shape)

# 데이터 균형 확인
print(data2.groupby('poison').agg(len))
print(data2.groupby('treat').agg(len))
print(data2.groupby(['poison', 'treat']).agg(len))
# 모든 집단별 표본 수가 동일하므로 균형설계가 잘 되었다 라고 할 수 있다.
result2 = ols('time ~ C(poison)* C(treat)', data=data2).fit()
print(anova_lm(result2))

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tkResult1 = pairwise_tukeyhsd(endog=data2.time, groups=data2.poison)
print(tkResult1)
tkResult2 = pairwise_tukeyhsd(endog=data2.time, groups=data2.poison)
print(tkResult2)

tkResult1.plot_simultaneous(xlabel='mean', ylabel='posion')
tkResult2.plot_simultaneous(xlabel='mean', ylabel='treat')
plt.show()
plt.close()