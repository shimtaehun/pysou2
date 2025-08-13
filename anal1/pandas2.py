# 재색인
from pandas import Series,DataFrame
import numpy as np

# Series의 재색인
data = Series([1,3,2], index=(1,4,2))
print(data)
data2 = data.reindex((1, 2, 4))
print(data2)

print('재색인할 떄 값 채우기 ---')
data3 = data2.reindex([0,1,2,3,4,5])
print(data3)
# 대응값이 없는(NaN) 인덱스는 결측값인데 777로 채우기
data3 = data2.reindex([0,1,2,3,4,5], fill_value=777)
print(data3)
print()
data3 = data2.reindex([0,1,2,3,4,5], method='ffill')
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method='pad')
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method='bfill')
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method='backfill')
print(data3)

# bool 처리, 슬라이싱 관련 method : loc(), iloc()
# 복수 인덱싱 : loc() 라벨지원, iloc() 숫자지원
df = DataFrame(np.arange(12).reshape(4, 3), index=['1월', '2월', '3월', '4월'],
                columns=['강남', '강북', '서초'])
print(df)
print(df['강남'])
print(df['강남'] > 3)
print(df[df['강남'] > 3])
print()
print(df.loc[:'2월'])
print(df.loc[:'2월', ['서초']])
print()
print(df.iloc[2])
print(df.iloc[2, :])

print(df.iloc[:3, 2]), type(df.iloc[:3, 2])
print()
print(df.iloc[:3, 1:3])