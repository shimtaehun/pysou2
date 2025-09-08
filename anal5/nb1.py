import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print(df.head(2))
print(df.info())
x = df[['MinTemp', 'MaxTemp', 'Rainfall']]
label = df['RainTomorrow'].map({'Yes':1, 'NO':0})
print(x[:3])
print(label[:3])

train_x, test_x, train_y, test_y =train_test_split(x, label, test_size=0.3, random_state=0)

gmodel = GaussianNB()
gmodel.fit(train_x, train_y)

pred = gmodel.predict(test_x)
print('예측값: ', pred[:10])
print('실제값: ', test_y[:10].values)
acc = sum(test_y == pred) / len(pred)
print('정확도 1: ', acc)
print('정확도2: ', accuracy_score(test_y, pred))
