import pandas as pd

df = pd.read_csv('./mir_train2.csv')

df['id'] = df['id'].apply(lambda x: x-300000)

df['path'] = df['id'].apply(lambda x: f'{x//10000}/{x}.jpg')

print(df.head())

df.to_csv('./mir_train2.csv', index=False)