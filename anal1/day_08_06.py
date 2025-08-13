import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
# pandas 문제 5-1)
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]

df['Age type'] = pd.cut(df['Age'], bins = bins, labels = labels)
print(df)

# pandas 문제 5-2)
print(df.pivot_table(index = ['Sex'], columns = 'Pclass', values = 'Survived',aggfunc = 'mean').round(2))

