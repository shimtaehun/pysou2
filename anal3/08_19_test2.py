import numpy as np
import pandas as pd
import scipy.stats as stats
import MySQLdb
import csv
from scipy import stats
import numpy as np
import MySQLdb
import matplotlib.pyplot as plt
import random
random.seed(42)

blue_data = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red_data = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

print('red_data: pvalue: ', stats.shapiro(red_data).pvalue)
print('blue_data: pvalue: ', stats.shapiro(blue_data).pvalue)
two_sample = stats.ttest_ind(red_data, blue_data)
print(two_sample)
# 포장지에 따른 매출액의 차이가 빨간색 일때의 값이 매출이 더 높다고 볼 수 있다.

boy_data = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
girl_data = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

print('boy_data: pvalue: ', stats.shapiro(boy_data).pvalue)
print('girl_data: pvalue: ', stats.shapiro(girl_data).pvalue)
print('등분산성: ', stats.levene(boy_data, girl_data).pvalue)
print(stats.ttest_ind(boy_data, girl_data, equal_var=True))

conn = MySQLdb.connect(
    host='localhost',
    user='root',
    passwd='1234',
    db='mydb',
    port=3306,
    charset='utf8'
)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    # 총무부 연봉 추출
    sql = """
        SELECT b.busername, j.jikwonpay
        FROM jikwon AS j
        JOIN buser AS b ON j.busernum = b.buserno
        WHERE b.busername ='총무부', '영업부';
    """
    cursor.execute(sql)
    results = cursor.fetchall()
    총무부 = [result[0] for result in results]
    print('총무부 연봉: ', 총무부)
    cursor.execute(sql)

    results = cursor.fetchall()
    영업부 = [result[0] for result in results]
    print('영업부 연봉: ', 영업부)

except Exception as e:
    print('처리 오류: ', e)
finally:
    conn.close()

plt.boxplot([총무부, 영업부], meanline=True, showmeans=True)
plt.show()
plt.close()

print(np.mean(총무부), ' ', np.mean(영업부))   # 5414.28   4908.33
two_sample = stats.ttest_ind(총무부, 영업부)
print(two_sample)