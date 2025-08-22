import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import seaborn as sns

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv')
# print(data.head(3))
data = data[['tv', 'radio', 'newspaper']]
corr = data.corr()
# print(corr)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()