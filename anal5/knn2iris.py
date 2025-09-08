# 필요한 라이브러리들을 불러옵니다.
from sklearn import datasets  # 사이킷런에 내장된 예제 데이터셋을 사용하기 위해 import 합니다.
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델을 사용하기 위해 import 합니다.
import numpy as np  # 수치 계산, 특히 배열과 행렬 연산을 위해 numpy를 np라는 별칭으로 import 합니다.
import pandas as pd  # 데이터 분석 및 조작을 위해 pandas를 pd라는 별칭으로 import 합니다. (여기서는 crosstab에 사용)
from sklearn.metrics import accuracy_score  # 모델의 분류 정확도를 평가하기 위해 import 합니다.
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 테스트용으로 나누기 위해 import 합니다.
from sklearn.preprocessing import StandardScaler  # 데이터 표준화를 위해 import 합니다. (주석 처리된 부분에서 사용)
import matplotlib.pyplot as plt  # 데이터 시각화를 위해 matplotlib.pyplot을 plt라는 별칭으로 import 합니다.
plt.rc('font', family='malgun gothic')  # 그래프에서 한글 폰트가 깨지지 않도록 '맑은 고딕'으로 설정합니다.
plt.rcParams['axes.unicode_minus'] = False  # 그래프에서 마이너스(-) 부호가 깨지지 않도록 설정합니다.
from matplotlib.colors import ListedColormap  # 시각화 시 색상 맵을 사용자 정의하기 위해 import 합니다. (이 코드에서는 직접 사용되지 않음)
import pickle  # 학습된 모델을 파일로 저장하거나 불러오기 위해 import 합니다.

# 로지스틱 회귀 모델 클래스를 다시 import 합니다. (위에서 이미 import 했으므로 중복되지만 코드 실행에는 문제 없습니다.)
# 로지스틱 회귀는 다중 클래스 분류(결과 값이 여러 개인 경우)를 지원합니다.
from sklearn.linear_model import LogisticRegression

# 사이킷런에 내장된 붓꽃(iris) 데이터셋을 불러옵니다.
iris = datasets.load_iris()
# print(iris['data'])  # 붓꽃 데이터의 독립 변수(특성) 데이터를 출력하는 코드입니다. (주석 처리됨)

# 붓꽃 데이터의 세 번째 특성(petal length)과 네 번째 특성(petal width) 간의 상관계수를 계산하고 출력합니다.
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))  # 출력 결과: 0.96286, 매우 높은 양의 상관관계를 보입니다.

# 독립 변수(x)로 붓꽃 데이터의 세 번째와 네 번째 특성(꽃잎 길이, 꽃잎 너비)만 선택합니다.
x = iris.data[:, [2, 3]]  # x는 행렬(matrix) 형태가 됩니다.

# 종속 변수(y)로 붓꽃의 품종(target)을 사용합니다.
y = iris.target  # y는 벡터(vector) 형태가 됩니다.

# x와 y 데이터의 처음 3개 행을 출력하여 데이터 형태를 확인합니다.
print('x: ', x[:3])
print('y: ', y[:3], set(y))  # set(y)를 통해 y에 어떤 클래스(품종)들이 있는지 확인합니다. {0, 1, 2}

# 전체 데이터를 훈련용(train) 데이터와 테스트용(test) 데이터로 7:3 비율로 분할합니다.
# random_state=0은 코드를 다시 실행해도 항상 동일한 방식으로 데이터가 나뉘도록 난수 시드를 고정하는 역할을 합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 분할된 데이터들의 크기(shape)를 출력하여 확인합니다. (행, 열) 순서로 표시됩니다.
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

"""
# 이 블록은 데이터 표준화(Scaling)를 수행하는 예시 코드입니다. (현재는 주석 처리되어 실행되지 않음)
# 표준화는 각 특성의 데이터 범위를 비슷하게 만들어 모델의 학습 안정성과 속도를 향상시키는 효과가 있습니다.
print('-'*100)
#-------------------------------------------------------------------------------------------
# Scaling(데이터 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버플로우/언더플로우 방지 효과)
print(x_train[:3])  # 표준화 전의 훈련 데이터를 출력합니다.
sc = StandardScaler()  # StandardScaler 객체를 생성합니다.
sc.fit(x_train); sc.fit(x_test)  # 훈련 데이터와 테스트 데이터의 평균과 표준편차를 계산합니다. (원래는 훈련 데이터로만 fit해야 합니다)
x_train = sc.transform(x_train)  # 계산된 통계치를 이용해 훈련 데이터를 표준화합니다.
x_test = sc.transform(x_test)  # 훈련 데이터의 기준으로 테스트 데이터를 표준화합니다.
print(x_train[:3])  # 표준화 후의 훈련 데이터를 출력합니다.
# 스케일링 원복: 표준화된 데이터를 원래의 값으로 되돌립니다.
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3]) # 원복된 데이터를 출력합니다.
#--------------------------------------------------------------------------------------------
"""

# 로지스틱 회귀 분류 모델을 생성합니다.
# C: 규제(regularization) 강도를 조절하는 파라미터입니다. 값이 작을수록 규제가 강해져 과적합을 방지하는 효과가 있습니다.
# random_state=0: 모델 내부의 무작위성을 제어하여 실행할 때마다 동일한 결과를 얻기 위해 설정합니다.
# model = LogisticRegression(C = 0.1, random_state=0, verbose=0)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
print('여기: ', model)  # 생성된 모델의 정보를 출력합니다.
# 훈련 데이터(x_train, y_train)를 사용하여 모델을 학습시킵니다.
model.fit(x_train, y_train)

# 학습된 모델을 사용하여 테스트 데이터(x_test)의 클래스(품종)를 예측합니다.
y_pred = model.predict(x_test)
print('예측값: ', y_pred)  # 모델이 예측한 값을 출력합니다.
print('실제값: ', y_test)  # 실제 정답 값을 출력하여 예측값과 비교합니다.

# 테스트 데이터 총 45개 중에서 모델이 잘못 예측한 개수를 계산하여 출력합니다.
print('총 갯수:%d, 오류수: %d'%(len(y_test), (y_test != y_pred).sum()))
print('-'*100)  # 결과 구분을 위해 선을 긋습니다.
# 모델의 분류 정확도를 확인하는 세 가지 방법을 보여줍니다.
print('분류정확도 확인1: ')     # 0.97778
# accuracy_score 함수를 사용하여 실제값(y_test)과 예측값(y_pred)을 비교하여 정확도를 계산합니다.
print('%.5f'%accuracy_score(y_test, y_pred))

print('분류정확도 확인2: ')
# pandas의 crosstab 함수를 이용해 혼동 행렬(confusion matrix)을 만듭니다.
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)  # 생성된 혼동 행렬을 출력합니다.
# 혼동 행렬의 대각선(정답을 맞춘 경우)의 합을 전체 데이터 수로 나누어 직접 정확도를 계산합니다.
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

print('분류정확도 확인3: ')
# 모델 객체에 내장된 score 메서드를 사용하여 테스트 데이터에 대한 정확도를 바로 계산합니다.
print('test: ', model.score(x_test, y_test))
# 훈련 데이터에 대한 정확도도 계산합니다. 이 값과 test 정확도의 차이가 크면 과적합(overfitting)을 의심할 수 있습니다.
print('train: ', model.score(x_train, y_train))

# 학습이 완료된 모델을 파일로 저장합니다.
# pickle.dump를 사용하여 'model' 객체를 'logimodel.sav'라는 파일에 바이너리 쓰기('wb') 모드로 저장합니다.
pickle.dump(model, open('logimodel.sav', 'wb'))

# 현재 메모리에 있는 model 변수를 삭제합니다.
del model

# 파일로 저장했던 모델을 다시 불러옵니다.
# 'logimodel.sav' 파일을 바이너리 읽기('rb') 모드로 열어 'read_model'이라는 변수에 로드합니다.
read_model = pickle.load(open('logimodel.sav', 'rb'))

# 새로운 데이터로 예측을 수행하기 위해 기존 테스트 데이터의 일부를 참고용으로 출력합니다.
print(x_test[:3])

# 예측해볼 새로운 데이터를 numpy 배열 형태로 생성합니다. (꽃잎 길이, 꽃잎 너비)
new_data = np.array([[5.1, 1.1], [1.1, 1.1], [6.1, 7.1]])

# 참고: 만약 모델을 학습시킬 때 데이터 표준화를 했다면, 새로운 데이터에도 동일한 표준화 작업을 적용해야 합니다.
# sc.fit(new_data); new_data = sc.transform(new_data) # (주석 처리됨)

# 불러온 모델(read_model)을 사용하여 새로운 데이터(new_data)의 품종을 예측합니다.
new_pred = read_model.predict(new_data)
print('예측 결과: ', new_pred)  # 예측된 클래스(0, 1, 2)를 출력합니다.

# predict_proba 메서드를 사용하여 새로운 데이터가 각 클래스(0, 1, 2)에 속할 확률을 출력합니다.
# 내부적으로 softmax 함수를 거친 결과값입니다.
print(read_model.predict_proba(new_data))
# 시각화
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                        np.arange(x2_min, x2_max, resulution)) 
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, \
                test_idx = range(100, 150), title='scikit-learn 제공')

# 트리 형태의 시각화
from sklearn import tree
from io import StringIO
import pydotplus
dot_data = StringIO()   # 파일 흉내를 내는 역할
tree.export_graphviz(read_model, out_file=dot_data,
                    feature_names=iris.feature_names[2:4])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('mytree.png')

from matplotlib.pyplot import imread
img = imread('mytree.png')
plt.imshow(img)
plt.show()
plt.close()