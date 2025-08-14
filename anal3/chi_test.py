# #카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
# 예제파일 : cleanDescriptive.csv
# 칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
# 조건 :  level, pass에 대해 NA가 있는 행은 제외한다.
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv")
data = data.dropna(subset=['level', 'pass'])
print(data.head(3))
# print(data[(data['level'] > 1.0) & (data['pass'] > 1.0)])
ctab = pd.crosstab(index=data['pass'], columns=data['level'])
ctab.columns = ["부모의 학력: 1.0", "부모의 학력:2.0", "부모의 학력:3.0"]
ctab.index = ["자녀의 대학 진학 1.0", "자녀의 대학 진학 2.0"]
print(ctab)
chi2, p, dof, _=stats.chi2_contingency(ctab)
msg = "test statics:{}, p-value:{}, df:{}"
print(msg.format(chi2, p, dof))

# p-value의 값이 0.251 이므로 부모의 학력 수ㅡ준과 자녀의 대학 진학 여부는 통계적으로 유의미한 관련이 있다고 볼 수 없다.

# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
# 그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
# 예제파일 : MariaDB의 jikwon table 
# jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
# jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
# 조건 : NA가 있는 행은 제외한다.

import MySQLdb

conn = MySQLdb.connect(
    host='localhost',
    user='root',
    passwd='1234',
    db='mydb',
    port=3306,
    charset='utf8'
)

sql ="""
    SELECT jikwonjik, jikwonpay
    FROM jikwon
"""
df = pd.read_sql(sql, conn)
print(df)
data = {'ijikwonjik':['이사': 1, '부장': 2, '과장': 3, '대리': 4, '사원': 5]
}

if 1000 < jikwonpay < 2999:
    return 1
elif 3000 < jikwonpay < 4999:
    return 2
elif 5000 < jikwonpay < 6999:
    return 3
else:
    return 4
chi2, p, dof, expected = stats.chi2_contingency(df)
print(chi2, p, dof, expected)