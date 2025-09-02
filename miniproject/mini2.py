import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False


data = pd.read_csv(f'miniproject\marketing_campaign.csv', sep='\t')
# print(data)

data = data.drop(['ID','Dt_Customer', 'Z_CostContact', 'Z_Revenue'], axis=1)
data['Education'] = data['Education'].map({'Basic':1, '2n Cycle':2, 'Graduation':3, 'Master':4, 'PhD':5})
data['Marital_Status'] = data['Marital_Status'].map({'Single':1, '2n Together':2, 'Married':3, 'Divorced':4, 'Widow':5, 'Alone':6, 'Absurd':7, 'YOLO':8})

data.columns = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
                'Teenhome',  'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                'AcceptedCmp2', 'Complain', 'Response']

sns.heatmap(data.corr(), annot=True, cmap='Blues')
plt.show()
plt.close()