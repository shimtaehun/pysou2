import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import f_oneway

data = pd.read_csv(f'miniproject\marketing_campaign.csv',sep='\t')
print(data)
income = 'Income'
wine = 'MntWines'
data['income_group'] = pd.qcut(data[income], q=4, labels=['low', 'mid_low', 'mid_high', 'high'])
group = data.groupby('income_group')[wine]
f_statistic, p_value = f_oneway(*group.apply(list))
print(f"F-statistic (F 통계량): {f_statistic:.4f}")
print(f"p_value (p_value): {p_value}")
print(income.summary()) 
# print('등분산성: ', stats.levene(income, wine).pvalue)

# print('정규성 검정: ', stats.shapiro(data.Income).pvalue)
# print('정규성 검정: ', stats.shapiro(data.MntWines).pvalue)