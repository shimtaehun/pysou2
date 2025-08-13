# file i/o
import pandas as pd

# df = pd.read_csv('anal1/ex1.csv')
df = pd.read_csv('anal1/ex1.csv', sep=',')
print(df,type(df))
print('*'*50)
print(df.info())
print()
df = pd.read_table('anal1/ex1.csv', sep=',')
print(df)
print(df,type(df))
print()
print()
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)
print(df)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                header=None, names = ['a', 'b','c', 'd','msg'], skiprows=1)
print(df)
print()
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt')
print(df)
print(df.info())
print()
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', sep='\s+')
print(df)
print()
print(df.info())
print()
df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt', 
                widths=(10, 3, 5), names=('date','name','price'), encoding='utf')
print(df)

print()
"""
url="https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4"
df = pd.read_html(url)
print(df)
print(f"총 {len(df)}개의 자료")
"""
# 대량의 데이터 파일을 읽는 경우 chunk 단위로 분리해 읽기 가능
# 1) 메모리 절약
# 2) 스트리밍 방식으로 순차적 처리 (로그 분석, 실시간 데이터, ML 데이터 처리 ...)
# 3) 분산 처리(batch)
import time
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'

n_rows = 10000

"""
data = {
    'id': range(1, n_rows + 1),
    'name': [f'Student_{i}' for i in range(1, n_rows + 1)],
    'score1':np.random.randint(50, 101, size=n_rows),
    'score2':np.random.randint(50, 101, size=n_rows)
}
df = pd.DataFrame(data)
print(df.head(3))
print(df.tail(3))
csv_path = 'students.csv'
df.to_csv(csv_path, index=False)
"""

# 작성된 csv 파일 사용 : 전체 한 방에 처리
start_all = time.time()
df = pd.read_csv('students.csv')
df_all = df
# print(df_all)
average_all_1 = df_all['score1'].mean()
average_all_2 = df_all['score2'].mean()
time_all = time.time() - start_all
print(time_all)

# chunk로 처리
chunk_size = 1000
total_score1 = 0
total_score2 = 0
total_count = 0

start_chunk = time.time()
start_chunk_total = time.time()
for i, chunk in enumerate(pd.read_csv('students.csv', chunksize=chunk_size)):
    start_chunk = time.time()
    # 청크 처리할 때 마다 첫번째 학생 정보 출력
    first_student = chunk.iloc[0]
    print(f"Chunk {i + 1} 첫번째 학생:id={first_student['id']}, 이름={first_student['name']},"
            f"score1={first_student['score1']}, score2={first_student['score2']}")
    total_score1 += chunk['score1'].sum()
    total_score2 += chunk['score2'].sum()
    total_count += len(chunk)
    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f"    처리 시간: {elapsed}초")

time_chunk_total = time.time() - start_chunk_total
average_chunk_1 = total_score1 / total_count    # score1 전체 평균
average_chunk_2 = total_score2 / total_count    

print('\n처리 결과 요약')
print(f'전체 학생 수: {total_count}')
print(f'score1 총합 : {total_score1}, score1 전체 평균: {average_all_1:.4f}')
print(f'score2 총합 : {total_score2}, score2 전체 평균: {average_all_2:.4f}')

print(f'전체 처리 시간: {time_all:.4f}초')
print(f'청크로 처리 한 경우 전체 소요 시간: {time_chunk_total:.4f}초')

# 시각화 
labels = ['전체 한번에 처리', '청크로 처리']
times = [time_all, time_chunk_total]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue', 'yellow'])

for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time_val:.4f}초',
            ha='center', va='bottom', fontsize=10)

plt.ylabel('처리시간(초)')
plt.title('전체 한번에 처리 VS 청크로 처리')
plt.grid(alpha = 0.5)
plt.tight_layout()
plt.show()

