#pandas 문제 1)
# a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. np.random.randn(9, 4)
# b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
# c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용

import numpy as np
import pandas as pd
from pandas import Series

data = np.random.randn(9,4)
print(data)
df = pd.DataFrame(data, columns=['No1','No2','No3','No4'])
print(df)

print('평균은:\n', np.mean(df[['No1','No2','No3','No4']], axis = 0))

# pandas 문제 2)
# a) DataFrame으로 위와 같은 자료를 만드시오. colume(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
# b) c row의 값을 가져오시오.
# c) a, d row들의 값을 가져오시오.
# d) numbers의 합을 구하시오.
# e) numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.
# f) floats 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5 결과가 나와야함
# g) names라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용.

data = [[10], [20], [30], [40]]
df1 = pd.DataFrame(data, columns = ['numbers'], index = ['a', 'b', 'c', 'd'])
print(df1)
print(df1.loc['c'])
print()
print(df1.loc[['a','d']])
print(df1.sum())
print(df1['numbers'] ** 2)
df1['floats'] = ['1.5', '2.5', '3.5', '4.5']
print(df1)
df2= Series(['길동', '오정', '팔계', '오공'], index = ['d', 'a', 'b', 'c'])
print(df2)

# pandas 문제 3)

# 1) 5 x 3 형태의 랜덤 정수형 DataFrame을 생성하시오. (범위: 1 이상 20 이하, 난수)
# 2) 생성된 DataFrame의 컬럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.
# 3) A 컬럼의 값이 10보다 큰 행만 출력하시오.
# 4) 새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.
# 5) 행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.
# 6) 아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.

# 1)
data = np.random.randint(1, 21, size = (5, 3))
print(data)
# 2)
df = pd.DataFrame(data, columns = ['A', 'B', 'C'], index = ['r1', 'r2', 'r3', 'r4', 'r5'])
print(df)
# 3)

# 4)
df['D'] = df['A'] + df['B']
print(df)
