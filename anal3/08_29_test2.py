import numpy as np
import pandas as pd
import scipy.stats as stats
import MySQLdb
import csv
from scipy import stats
import numpy as np
import MySQLdb

# 귀무가설: 각 부서별로 연봉의 차이가 없다.
# 대립가설: 각 부서별로 연봉의 차이가 있다.

conn = MySQLdb.connect(
    host='localhost',
    user='root',
    passwd='1234',
    db='mydb',
    port=3306,
    charset='utf8'
)

sql = """
    SELECT b.busername, j.jikwonpay
    FROM jikwon AS j
    JOIN buser AS b ON j.busernum = b.buserno
    WHERE b.busername IN ('총무부', '영업부', '전산부', '관리부');
    """
df = pd.read_sql(sql, conn)
df.columns = ['부서명', '연봉']
print(df)
buser_group = df.groupby('부서명')['연봉'].mean()
print(buser_group)

gwanli = df[df['부서명'] == '관리부']['연봉']
yeongup = df[df['부서명'] == '영업부']['연봉']
jeonsan = df[df['부서명'] == '전산부']['연봉']
chongmu = df[df['부서명'] == '총무부']['연봉']

f_statistic, p_value = stats.f_oneway(gwanli, yeongup, jeonsan, chongmu)
print('f_statistic:', f_statistic)
print('p_value: ', p_value)

# p_value: 0.7454421884076983 > 0.05 이므로 귀무가설 채택