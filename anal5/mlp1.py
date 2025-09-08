# 단층 신경망(MLP)

# 단층 신경망으로 논리회로 분류
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(feature)
# label = np.array([0, 0, 0, 1])  # and
# label = np.array([0, 1, 1, 1])  # or
label = np.array([0, 1, 1, 0])    # xor

ml = MLPClassifier(hidden_layer_sizes=(10,10,10), solver='adam', learning_rate_init=0.01).fit(feature, label)   # eta0: learning_rate(학습률)

print(ml)
pred = ml.predict(feature)
print('pred', pred)
print('acc: ', accuracy_score(label, pred))