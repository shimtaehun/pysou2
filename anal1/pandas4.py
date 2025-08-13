# 데이터프레임 병합
import pandas as pd # 데이터 분석을 위한 pandas 라이브러리를 불러옵니다.
import numpy as np # 수치 연산을 위한 numpy 라이브러리를 불러옵니다. (이 코드에서는 직접 사용되지 않음)

# 첫 번째 데이터프레임(df1)을 생성합니다.
# 'data1' 열은 0부터 6까지의 숫자를, 'key' 열은 문자열 리스트를 가집니다.
df1 = pd.DataFrame({'data1': range(7), 'key': ['b', 'b', 'b', 'c', 'a', 'a','b']})
print(df1) # 생성된 df1의 내용을 출력합니다.

# 두 번째 데이터프레임(df2)을 생성합니다.
# 'key' 열은 병합의 기준이 되며, 'data2' 열은 0부터 2까지의 숫자를 가집니다.
df2 = pd.DataFrame({'key': ['a','b','d'], 'data2':range(3)})
print(df2) # 생성된 df2의 내용을 출력합니다.

print()

# pd.merge를 사용하여 df1과 df2를 병합하고 결과를 출력합니다.
print(pd.merge(df1, df2, on='key'))     # 'key'열을 기준으로 병합합니다. how의 기본값은 'inner'입니다.
print()
print(pd.merge(df1, df2, on='key', how='inner')) # 두 데이터프레임에 공통으로 존재하는 'key'를 기준으로 병합합니다. (교집합)
print()
print(pd.merge(df1, df2, on='key', how='outer')) # 두 데이터프레임의 모든 'key'를 기준으로 병합합니다. 한쪽에만 데이터가 있는 경우 NaN으로 채워집니다. (합집합)
print()
print(pd.merge(df1, df2, on='key', how='left')) # 왼쪽 데이터프레임(df1)의 'key'를 모두 유지하면서 병합합니다. df2에 해당 key가 없으면 NaN으로 채워집니다.
print()
print(pd.merge(df1, df2, on='key', how='right')) # 오른쪽 데이터프레임(df2)의 'key'를 모두 유지하면서 병합합니다. df1에 해당 key가 없으면 NaN으로 채워집니다.
print()
print('--- 공통 칼럼 이름이 없는 경우 병합 ---')
# 세 번째 데이터프레임(df3)을 생성합니다. 병합 기준 열의 이름이 'key2'로 df1의 'key'와 다릅니다.
df3 = pd.DataFrame({'key2': ['a', 'b', 'c'], 'data2': range(3)})
print(df1) # df1 출력
print(df3) # df3 출력
print()
# 'key'와 'key2'를 기준으로 병합합니다. left_on은 왼쪽 DF(df1)의 기준 열, right_on은 오른쪽 DF(df3)의 기준 열을 지정합니다.
print(pd.merge(df1, df3, left_on='key', right_on='key2')) 

print('\n--- 데이터프레임 이어붙이기 (concat) ---')
# concat을 사용하여 두 데이터프레임을 옆으로(axis=1) 이어붙입니다. 인덱스를 기준으로 정렬되며, 공통 열 이름이 없으면 그대로 합쳐집니다.
print(pd.concat([df1, df3], axis=1))

# 여러 Series를 이어붙이기 위한 샘플 데이터(s1, s2, s3)를 생성합니다.
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
# s1, s2, s3를 위아래로(axis=0) 이어붙입니다. axis=0은 기본값이므로 생략 가능합니다.
print(pd.concat([s1, s2, s3], axis=0))

print('그룹화 : pivot_table')
data = {
    'city': ['강남', '강북', '강남', '강북'],
    'year': [2000, 2001, 2002, 2003],
    'pop': [3.3, 2.5, 3.0, 2.0]
}

df= pd.DataFrame(data)
print(df) # pivot 및 pivot_table 예제에 사용할 데이터프레임을 출력합니다.
print(df.pivot(index='city', columns='year', values='pop'))
print()
print(df.set_index(['city', 'year']).unstack())
print()
print(df.describe())
print('pivot_table: pivot과 groupby의 중간적 성격')
print(df)
print(df.pivot_table(index=['city']))
print(df.pivot_table(index=['city'], aggfunc='mean'))
print(df.pivot_table(index=['city','year'], aggfunc=[len, 'sum']))
print(df.pivot_table(index='city', values='pop', aggfunc='mean'))
print(df.pivot_table(index='city', values='pop', aggfunc='mean'))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], margins=True, fill_value=0))
print()
hap = df.groupby(['city'])
print(hap)
print(hap.sum())
print(df.groupby(['city']).sum())
print(df.groupby(['city', 'year']).mean())