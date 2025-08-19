import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

plt
# 귀무 : 새로운 백열전구의 평균 수명은 300시간이다.
# 대립 : 새로운 백열전구의 평균 수명은 300시간이 아니다.
data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
fdata = pd.DataFrame(data, columns=['수명'])

print(fdata.mean())
print('정규성 검정:',stats.shapiro(fdata.수명))


res = stats.ttest_1samp(fdata.수명, popmean=300)
print('statistic:%.5f,pvalue:%.5f'%res)

# sns.displot(fdata.수명, kde=True)
# plt.show()
# plt.close()

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
data['time'] = pd.to_numeric(data['time'], errors='coerce') # 변환 할수 없는 데이터는 NaN 처리 여기서는 공백이 해당
df = data.dropna(axis=0) # NaN 행 삭제
print(df['time'].head(10))
print('정규성 검정:',stats.shapiro(df.time)) # pvalue=0.7242303336695732 0.05보다 크므로 정규성 만족

# 정규성을 만족하면 T-검정 을사용 그렇지 않으면 윌콕슨 검정 사용

# wilcox_res = wilcoxon(df.time - 5.2) # 평균 5.2과 비교
# print('wilcox_res: ',wilcox_res) # pvalue=0.00025724108542552436 0.05보다 작으므로 귀무가설 기각

res = stats.ttest_1samp(df.time, popmean=5.2) # pvalue:0.00014: 0.05보다 작으므로 귀무 가설 기각
print('statistic:%.5f,pvalue:%.5f'%res)

# 결론 A회사 노트북  평균 사용 시간은 5.2시간이 아니다.

sns.displot(df.time, kde=True)
plt.show()
plt.close()