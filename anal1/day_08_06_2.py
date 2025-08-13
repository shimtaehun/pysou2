import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/human.csv')
print(df)
print(df.columns)

df[' Group'] = df[' Group'].replace('NA', np.nan)
df[' Group'] = df[' Group'].str.strip(' ') 
print(df)
df_cleaned = df.dropna(subset=[' Group'])
print(df_cleaned)