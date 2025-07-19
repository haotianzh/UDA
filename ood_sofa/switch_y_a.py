import pandas as pd 

prefix = 'SOFA'

df = pd.read_csv(f'{prefix}.csv')
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df1['a'] = 1 - df1['a']
df2['y'] = 1 - df2['y']
df3['a'] = 1 - df3['a']
df3['y'] = 1 - df3['y']


df1.to_csv(f'{prefix}_switch_a.csv', index=None)
df2.to_csv(f'{prefix}_switch_y.csv', index=None)
df3.to_csv(f'{prefix}_switch_a_y.csv', index=None)