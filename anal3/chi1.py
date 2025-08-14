# 교차분석(카이제곱) 가설검정, cross-tabulation analysis
# 정식명칭 : pearsons's chi-square test <-- 두 불연속변수(범주형) 간의 상관관계를 측정하는 기법 
# 데이터를파악할 때 중심위치(평균)와 퍼짐정도(분산)가 중요한데
# 카이제곱은 바로 분산에 대한 분포다
# 범주형 자료를 대상으로 교차 빈도에 대한 검정통계량, 유의성을 검증해 주는 추론통계 기법.
# 유형은 두 가지
# 일원카이제곱 (변인: 단수) - 적합도, 선호도 검증 - 교차 분할표 사용 X
# 이원카이제곱 (변인: 복수) - 독립성, 동질성 검증 - 교차 분할표 사용 O      이 방식을 주로 사용
# 유의확률(P-value)에 의해 집단 간에 차이 여부를 가설로 검증

# 교차분석 흐름 이해용 : 수식((관측값 - 기대값)제곱을 기대값으로 나눈 전체합)에 의해 카이제곱 값 구하기, 함수로 구하기
import pandas as pd
data = pd.read_csv("pass_cross.csv")
print(data.head(3))
print(data.tail(3))
# 귀무가설(H0) : 공부하는 것과 합격여부는 관계가 없다.
# 대립가설(H1) : 공부하는 것과 합격여부는 관계가 있다.
print(data[(data['공부함'] == 1)& (data['합격'] == 1)].shape[0])        # 18명
print(data[(data['공부함'] == 1)& (data['불합격'] == 1)].shape[0])      # 7명

# 빈도표 
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
ctab.columns = ['합격', '불합격', '행합']
ctab.index = ['공부함', '공부안함', '열합']
print(ctab)

# 방법1 : 수식 사용
# 기대 도수? (각 행의 주변합) * (각 열의 주변합) / 총합
# chi2는 카이제곱
# chi2 = (18-15) ** 2 / 15 .....
chi2 = 3.0
# 임계값은 ? (카이제곱표를 사용하면 구할 수 있다.)
# 자유도(df) (행의갯수 - 1) * (열의갯수 - 1)  ==>  2 - 1 = 1
# 임계값(C.V) : 3.84
# 결론 : 카이제곱 검정 통계량 3.0은 C.V 3.84 보다 작음으로 귀무채택역 내에 있다.
# 그러므로 대립가설을 기각하고 귀무가설은 채택한다.
# 벼락치기 공부하는 것과 합격여부는 관계가 없다.

print()
# 방법2 : 함수 사용
import scipy.stats as stats
chi2, p, dof, expected = stats.chi2_contingency(ctab)
print(chi2, p, dof, expected)
# chi2 : 3.0, p: 0.5578
msg = "Test statistic : {} p-value : {}"
print(msg.format(chi2, p))
# 결론 : p-value(0.5578) > (유의수준) 0.05 이므로 귀무가설을 채택
# 새로운 주장을 위해 수집한 data는 (필연이 아니라) 우연히 발생한 자료라고 할 수 있다.

