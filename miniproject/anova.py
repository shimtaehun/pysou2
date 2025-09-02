# 소득분위에 따른 고기, 와인 소비액 차이가 있다/없다(ANOVA)
#  - h0 : 소득분위에 따른 고기, 와인 소비량 차이가 없다
#  - h1 : 소득분위에 따른 고기, 와인 소비량 차이가 있다

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import seaborn as sns

plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv(r'miniproject\filtered_MarketingData.csv')
data.dropna(subset=['Income'], inplace=True)
bins = [0, 11600, 47150, np.inf]
labels = ['저소득층', '중산층', '고소득층']
data['Income_Group'] = pd.cut(data['Income'], bins=bins, labels=labels, right=True, include_lowest=True)
print(data['Income_Group'])
for col in ['MntMeatProducts', 'MntWines']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치를 벗어나는 행을 식별합니다.
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    print(f"'{col}' 열에서 제거될 이상치 수: {len(outliers)}")
    
    # 이상치가 아닌 데이터만 남깁니다.
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
groups_meat = [data['MntMeatProducts'][data['Income_Group'] == q] for q in ['저소득층', '중산층', '고소득층']]
groups_wines = [data['MntWines'][data['Income_Group'] == q] for q in ['저소득층', '중산층', '고소득층']]
f_stat_meat, p_value_meat = stats.f_oneway(*groups_meat)
f_stat_wines, p_value_wines = stats.f_oneway(*groups_wines)
print(f"고기 소비량(MntMeatProducts) 분석 결과: F-통계량 = {f_stat_meat:.4f}, p-value = {p_value_meat}")
print(f"와인 소비량(MntWines) 분석 결과: F-통계량 = {f_stat_wines:.4f}, p-value = {p_value_wines}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 왼쪽 그래프: 소득분위별 고기 소비량 박스플롯
sns.boxplot(x='Income_Group', y='MntMeatProducts', data=data, ax=axes[0])
axes[0].set_title('소득분위별 고기 소비량', fontsize=15)
axes[0].set_xlabel('소득분위', fontsize=12)
axes[0].set_ylabel('고기 소비량', fontsize=12)

# 오른쪽 그래프: 소득분위별 와인 소비량 박스플롯
sns.boxplot(x='Income_Group', y='MntWines', data=data, ax=axes[1])
axes[1].set_title('소득분위별 와인 소비량', fontsize=15)
axes[1].set_xlabel('소득분위', fontsize=12)
axes[1].set_ylabel('와인 소비량', fontsize=12)

plt.show()
plt.close()