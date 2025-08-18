# 이원카이제곱
# 동질성 :
# 검정 두 집단의 분포가 동일한가 다른 분포인가 를 ㅈ검증하는 방법이다 두 집단 이상에서 각
# 동일한가를 검정하게 된다 두 개 이상의 범주형 자료가 동일한 분포르ㅜㄹ 갖는

# 검정실습 1

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
print(data.head(3))
print(data['method'].unique())
print(set(data['survey']))

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족', '만족', '보통', '불만족', '매우불만족']
ctab.index = ['방법1', '방법2', '방법3']
print(ctab)

chi2, p, ddof, _=stats.chi2_contingency(ctab)
msg = 'test statistic:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))
# test statistic:6.544667820529891, p-value:0.5864574374550608, df:8
# 해석: 유의수준 0.05 < p-value:0.5864574374550608 이므로 귀무가설 채택

print('----------------------------------------')
# 검정 실습 2) 연령대별 sns 이용률의 동질성 검정
# 대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 sns 서비스들에 대해 이용
# 전략을 세우고자 한다. 연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자
# 귀무 가설: 연령대별로 SNS 서비스별 이용 현황은 동일하다
# 대립 가설: 연령대별로 SNS 서비스별 이용 현황은 동일하지 않다

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')
print(data2.head(3))
print(data2['age'].unique())        # [1 2 3]
print(data2['service'].unique())    # ['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data2['age'], columns=data2['service']) # , margins=True)
print(ctab2)
chi2, p, ddof, _=stats.chi2_contingency(ctab2)
msg = 'test statistic:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))
# test statistic:102.75202494484225, p-value:1.1679064204212775e-18, df:8
# 해석: 유의수준 0.05 > p-value:0.000 이므로 귀무가설 기각....

# 사실 위 데이터는 샘플데이터이다. 
# 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본을 추출해 처리해 보자.
sample_data = data2.sample(n=50, replace=True, random_state=1)
print(len(sample_data))
ctab3 = pd.crosstab(index=sample_data['age'], columns=sample_data['service']) # , margins=True)
print(ctab3)
chi2, p, ddof, _=stats.chi2_contingency(ctab3)
msg = 'test statistic:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))
