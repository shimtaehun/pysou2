# Local Database 연동 후 자료를 읽어 DataFrame에 저장
import sqlite3

sql = "create table if not exists test(product varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect(':memory:')      # 'testdv'
conn.execute(sql)
conn.commit()   

# 한 개씩 추가
stmt = "insert into test values(?, ?, ?, ?)"     # 시큐어 코딩에 위반
data1 = ('mouse', 'samsung', 12.5, 5000)        # (4)는 튜플 X (4, )는 튜플 O
conn.execute(stmt, data1)
data2 = ('mouse', 'samsung', 15.5, 8000)        # (4)는 튜플 X (4, )는 튜플 O
conn.execute(stmt, data2)

# 복수 개 추가
datas = [('mouse3', 'lg', 22.5,15000), ('mouse4','lg',25.5,15500)]
conn.executemany(stmt, datas)

cursor = conn.execute("select * from test")
rows = cursor.fetchall()
print(rows[0], ' ', rows[1], rows[0][0])
for a in rows:
    print(a)

import pandas as pd
df = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df)
# print(df.to_html())
print()
df2 = pd.read_sql("select * from test", conn)
print(df2)
print()
pdata = {
    'product':['연필','볼펜','지우개'],
    'maker':['동아','모나미','모나미'],
    'weight':[1.5, 5.5, 10.0],
    'price':[500, 1000, 1500]
}
frame = pd.DataFrame(pdata)
# print(frame)
frame.to_sql("test", conn, if_exists='append', index=False)

print('------------------------------------------------------')
df3 = pd.read_sql("select product, maker 메이커, price 가격, weight as 무게 from test", conn)
print(df3)

cursor.close()
conn.close()