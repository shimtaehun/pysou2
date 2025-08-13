# iris(붓꽃) dataset : 꽃받침과 꽃잎의 너비와 길이로 꽃의 종류를 3가지로 구분해 놓은 데이터
# 각 그룹당 50개, 총 150개 데이터
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv')
print(iris_data.info())
print(iris_data.head(3))
print(iris_data.tail(3))

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.xlabel('sepal.Lenght')
plt.ylabel('petal.Lenght')
plt.show()

print()
print(iris_data['Species'].unique())
print(set(iris_data['Species']))

cols = []
for s in iris_data['Species']:
    choice = 0
    if s == 'setosa':
        choice = 1
    elif s == 'versicolor':
        choice = 2
    elif s == 'virginica':
        choice = 3
    cols.append(choice)
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('Types of Flowers')
plt.show()


# 데이터 분포와 산점도 그래프 출럭
iris_col = iris_data.loc[:, 'Sepal.Length': 'Petal.Length']
print(iris_col)

from pandas.plotting import scatter_matrix      # pandas의 시각화 기능 활용
scatter_matrix(iris_col, diagonal='kde')
plt.show()

# seaborn 기능 사용
import seaborn as sns
sns.pairplot(iris_data, hue='Species', height=2)
plt.show()

# rug plot
x = iris_data['Sepal.Length'].values
sns.rugplot(x)
plt.show()

# kernel density
sns.kdeplot(x)
plt.show()