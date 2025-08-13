# pip install MySQLClient

import MySQLdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 표시 깨짐 방지
import sys
import pickle
import csv

# conn = MySQLdb.connect(
#     host='localhost',
#     user='root',
#     passwd='',
#     db='mydb',
#     port=3306,
#     charset='utf8'
# ) # 방법1

# 방법2 별도의 객체로 config 설정 (anal4_db/dbconfig.py에서 설정)
try:
    with open('mymaria.dat', mode='rb') as obj:
        config = pickle.load(obj)  # 바이너리 파일에서 config 객체 읽기
        print('config:', config)  # config 내용 출력
except Exception as e:
    print(f'읽기오류: {e}')
    sys.exit() # 프로그램 강제 종료

try:
    conn = MySQLdb.connect(**config) # 방법2 일때 conn 연결하는 부분
    cursor = conn.cursor()  # 커서 생성. 방법1, 2 모두 동일
    sql = """
    SELECT jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay 
    FROM jikwon inner join buser
    on jikwon.busernum = buser.buserno
    """
    cursor.execute(sql)

    # 출력1 : console - 연습용, 테스트용
    # for (a,b,c,d,e,f) in cursor:
        # print(f'jikwon: {a}, jikwonname: {b}, busername: {c}, jikwonjik: {d}, jikwongen: {e}, jikwonpay: {f}')

    for (jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay) in cursor:
        print(f'jikwonno: {jikwonno}, jikwonname: {jikwonname}, busername: {busername}, '
            f'jikwonjik: {jikwonjik}, jikwongen: {jikwongen}, jikwonpay: {jikwonpay}') # 가독성을 위해서는 이 방법이 더 좋음
    print('--' * 30)
    # 출력2 : dataframe
    df1 = pd.DataFrame(cursor.fetchall(), 
                    columns=['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])
    print('df1:\n', df1.head(3))
    print('--' * 30)

    # 출력3 : csv 파일 - 저장시 자주 사용
    with open('anal4_db/jik_data.csv', mode='w', encoding='utf-8', newline='') as fobj:
        writer = csv.writer(fobj)
        # writer.writerow(['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])  # 헤더 작성
        for row in cursor:
            writer.writerow(row)  # 각 행을 CSV 파일에 작성
    # CSV 파일 읽어 dataframe으로 변환
    df2 = pd.read_csv('anal4_db/jik_data.csv', header=None, 
                    names=['번호', '이름', '부서명', '직급', '연봉'])
    print('CSV 파일에서 읽은 데이터:\n', df2.head(3))

    print('\nDB의 자료를 pandas의 sql 처리 기능으로 읽기')
    df = pd.read_sql(sql, conn)  # SQL 쿼리로 DataFrame 생성
    df.columns = ['번호', '이름', '부서명', '직급', '성별', '연봉']  # 컬럼 이름 변경
    print('df:\n', df.head(3))  # DataFrame 출력

    print('\nDB의 자료를 DataFrame으로 읽었으므로 pandas의 기능을 적용 가능')
    print('건수:\n', len(df))  # DataFrame의 행 수
    print('이름 갯수:\n', df['이름'].count())  # 이름 컬럼의 행 수
    print('직급별 건수:\n', df['직급'].value_counts())  # 직급별 건수
    print('연봉 평균:\n', df.loc[:, '연봉'].mean())  # 연봉 평균 
    print()

    ctab = pd.crosstab(df['성별'], df['직급'], margins=True)  # 성별과 직급의 교차표
    # print('성별과 직급의 교차표:\n', ctab.to_html())  # HTML로 출력

    # 시각화 - 직급별 연봉 평균 - pie 차트
    jik_ypay = df.groupby('직급')['연봉'].mean()
    print('직급별 연봉 평균:\n', jik_ypay)
    print('index:\n', jik_ypay.index)
    print('values:\n', jik_ypay.values)

    plt.pie(jik_ypay, explode=(0.2, 0, 0, 0.3, 0),
            labels=jik_ypay.index,
            shadow=True,
            counterclock=False)  # 파이 차트
    plt.title('직급별 연봉 평균')
    plt.show()

except Exception as e:
    print(f'Error: {e}')