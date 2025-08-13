# a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
# - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
# - DataFrame의 자료를 파일로 저장
# - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
# - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
# - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
# - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성

import pandas as pd
import numpy as np
import mariadb
import sys
import pickle
import matplotlib.pyplot as plt

# 한글 폰트 및 마이너스 부호 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 데이터베이스 접속 정보 읽기 ---
try:
    with open('./mymaria.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print(f'설정 파일 읽기 오류: {e}')
    sys.exit()

# --- 2. MariaDB 연결 및 데이터 처리 ---
try:
    # MariaDB 연결
    conn = mariadb.connect(**config)
    cursor = conn.cursor()

    # --- 3. 사번, 이름, 부서명, 연봉, 직급으로 DataFrame 작성 ---
    print('--- 직원 정보 DataFrame ---')
    sql = """
        SELECT jikwon_no, jikwon_name, buser_name, jikwon_pay, jikwon_jik
        FROM jikwon
        JOIN buser ON jikwon.buser_num = buser.buser_no
    """
    df = pd.read_sql(sql, conn)
    df.columns = ['사번', '이름', '부서명', '연봉', '직급']
    print(df.head(3))

    # --- 4. DataFrame을 파일로 저장 ---
    # CSV 파일로 저장. 인덱스는 저장하지 않음.
    df.to_csv('jikwon_data.csv', index=False)
    print('\nDataFrame을 jikwon_data.csv 파일로 저장했습니다.')

    # --- 5. 부서명별 연봉 분석 ---
    print('\n--- 부서명별 연봉 (합, 최대, 최소) ---')
    buser_pay = df.groupby('부서명')['연봉'].agg(['sum', 'max', 'min'])
    print(buser_pay)

    # --- 6. 부서명, 직급별 교차 테이블 작성 ---
    print('\n--- 부서-직급 교차표 ---')
    cross_table = pd.crosstab(df['부서명'], df['직급'])
    print(cross_table)

    # --- 7. 직원별 담당 고객자료 출력 ---
    print('\n--- 직원별 담당 고객 정보 ---')
    sql_gogek = """
        SELECT jikwon_name, gogek_no, gogek_name, gogek_junhwa
        FROM jikwon
        LEFT JOIN gogek ON jikwon.jikwon_no = gogek.gogek_damsano
        ORDER BY jikwon_name
    """
    cursor.execute(sql_gogek)
    gogek_data = cursor.fetchall()
    
    # 모든 직원의 이름을 먼저 추출
    all_jikwon_names = df['이름'].unique()
    
    # 고객이 있는 직원 정보만 딕셔너리로 정리
    jikwon_gogek_map = {}
    for name, no, gname, tel in gogek_data:
        if no is not None: # 담당 고객이 있는 경우
            if name not in jikwon_gogek_map:
                jikwon_gogek_map[name] = []
            jikwon_gogek_map[name].append(f"  - 고객번호:{no}, 고객명:{gname}, 전화:{tel}")

    # 모든 직원을 순회하며 출력
    for name in all_jikwon_names:
        print(f"직원: {name}")
        if name in jikwon_gogek_map:
            for customer_info in jikwon_gogek_map[name]:
                print(customer_info)
        else:
            print("  담당 고객 X")


    # --- 8. 부서명별 연봉 평균 가로 막대 그래프 ---
    print('\n부서별 평균 연봉 그래프를 생성합니다.')
    avg_pay_by_buser = df.groupby('부서명')['연봉'].mean()
    
    plt.figure(figsize=(10, 6))
    avg_pay_by_buser.plot(kind='barh', title='부서별 평균 연봉', grid=True)
    plt.xlabel('평균 연봉')
    plt.ylabel('부서명')
    plt.show()

# try 블록에 대한 except와 finally 블록
except Exception as e:
    print(f'처리 중 오류 발생: {e}')
finally:
    # conn 변수가 생성되었는지 확인 후 연결을 닫음
    if 'conn' in locals() and conn:
        conn.close()
        print('\nMariaDB 연결이 해제되었습니다.')
