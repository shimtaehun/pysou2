import numpy as np

# 최소제곱법(ols)로 y = wx + b 형태의 추세식 파라미터와 w와 b의 값 구하기

class MySimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit (self, x:np.ndarray, y:np.ndarray):
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numeratir = np.sum((x - x_mean)*(y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        self.w = numeratir / denominator
        self.b = y_mean - (self.w * x_mean)
        # print(self.w)   # y = wx + b에서 w의 값
        # print(self.b)   # y = wx + b에서 b의 값
    def predict(self, x:np.ndarray):
        return self.w*x + self.b

    

def main():
    # 임의의 성인 남성 10명의 키, 몸무게 만들기
    x_heights = np.random.normal(175, 5, 10)
    y_weights = np.random.normal(70, 10, 10)
    
    model = MySimpleLinearRegression()
    model.fit(x_heights, y_weights)

    print('w: ', model.w)
    print('b: ', model.b)

    y_pred = model.predict(x_heights)
    
    print('실제 몸무게와 예측 몸무게 비교')
    for i in range(len(x_heights)):
        print(f"키: {x_heights[i]:.2f} | 실제 몸무게: {y_weights[i]:.2f} | 예측 몸무게: {y_pred[i]:.2f}")
main()