# 집단 차이분석: 평균 또는 비율 차이를 분석:
# T test 와 ANOVA 의 차이.

# 핵심 아이디어: 
# 집단 평균차이(분자)와 집단 내 변동성(표준편차, 표준오차 등, 분모)을 비교하여,
# 차이가 데이터의 불확실성(변동성)에 비해 얼마나 큰지를 계산한다.
# t분포는 표본 평균을 이용해 정규분포의 평균을  해석할 때 많이 사용한다.
# 대개의 경우 표본의 크기는 30개 이하일 때 t 분포를 따른다.
# t검정은 '두개 이하 집단의 평균의 차이가 우연에 의한 것인지 통계적으로 유의한 차이를 판단한다.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 실습1 - 어느 남성 집단의 평균키 검정
# 귀무: 집단의 평균 키가 177이다. (모수)
# 대립: 집단의 평균 키가 177이 아니다.
one_sample = [167.0, 182.7, 160.6, 176.8, 185.0]
print(np.array(one_sample).mean())  # 174.42
# 177.0과 174.42 는 평균의 차이가 있느냐?
result = stats.ttest_1samp(one_sample, popmean = 177)
print('statistic:%.5f, pvalue:%.5f'%result)
# statistic:-0.55499, pvalue:0.60847
# pvalue:0.60847 > 0.05 이므로 귀무가설 채택.
# plt.boxplot(one_sample)

# sns.displot(one_sample, bins = 10, kde=True, color='blue')
# plt.xlabel('data')
# plt.ylabel('value')
# plt.show()
# plt.close()

print('-'*100)
# 실습2 - 단일 모집단의 평균에 대한 가설검정    (one samples t-test)
# 중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 국어 편균검정   student
# 귀무: 학생들의 국어 점수의 평균은 80이다.
# 대립: 학생들의 국어 점수의 평균은 80이 아니다.
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
print(data.head(3))
print(data.describe())

# 정규성 검정: one-sample t-test는 옵션
print('정규성 검정: ', stats.shapiro(data.국어))    # pvalue(0.01295975332132026) < 0.05보다 작으므로 정규성 만족 못함
# 정규성 위배는 데이터 재가공 추천, wilcoxon Signed-rank test는 정규성을 가정하지 않음
from scipy.stats import wilcoxon
wilcox_res = wilcoxon(data.국어 - 80)   # 평균 80과 비교
print('wilcox_res: ', wilcox_res)
#  pvalue=np.float64(0.39777620658898905) > 0.05 이므로 귀무가설 채택
res = stats.ttest_1samp(data.국어, popmean=80)
print('statistic:%.5f, pvalue:%.5f'%res)
# statistic:-1.33218, pvalue:0.19856 > 0.05 이므로 귀무가설 채택

# 해석: 정규성은 부족하지만 t-test와 wilcoxon은 같은 결과를 얻었다. 표본수가 커지면 결과는 달라질 수 있다.
# 정규성 위배가 있어도 t-test결과는 신뢰 할 수 있다.
print('-'*30)
# 실습 3)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자
# 귀무: 여아 신생아의 몸무게는 평균이 2800(g)이다
# 대립: 여아 신생아의 몸무게는 평균이 2800(g) 보타 크다.
data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv')
print(data2.head(3))
print(data2.describe())
fdata = data2[data2.gender == 1]
print(fdata.head(2))
print(len(fdata))
print(np.mean(fdata.weight), ' ', np.std(fdata.weight))
# 3132 vs 2800 둘 사이는 평균에 차이가 있는가?

# 정규성 검정 (하나의 집단일 때는 option)
print(stats.shapiro(fdata.iloc[:, 2]))  # p값 0.01798 < 0.05 정규성 위배
# 정규성 시각화
# 1) histogram으로 확인 
sns.displot(fdata.weight, kde=True)
plt.show()
plt.close()

# 2) Q-Q plot으로 확인
stats.probplot(fdata.iloc[:, 2], plot=plt)
plt.show()
plt.close()

print()
wilcox_resBaby = wilcoxon(fdata.weight - 2800)   # 평균 2800과 비교
print('wilcox_res: ', wilcox_resBaby)
# pvalue: 0.03423 < 0.05 이므로 귀무기각.
print()
resBaby = stats.ttest_1samp(fdata.weight, popmean=2800)
print('statistic:%.5f, pvalue:%.5f'%resBaby)
# pvalye:0.03927 < 0.0a5 이므로 귀구기각.
# 즉, 여 신생아의 평균 체중은 2800g보다 증가하였다.
# 왜?