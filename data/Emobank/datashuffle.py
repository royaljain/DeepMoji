import numpy as np
import pandas as pd


df = pd.read_csv('Emobank.csv', sep='\t')


shuffle_indices = np.random.permutation(np.arange(len(df)))
train_len = int(len(shuffle_indices)*0.8)

train_indices = shuffle_indices[:train_len]
test_indices = shuffle_indices[train_len:]

print len(train_indices)
print len(test_indices)


df1 = df.iloc[train_indices]
df2 = df.iloc[test_indices]

print len(df1)
print len(df2)



df1.to_csv('EmoBankTrain.tsv', sep='\t')
df2.to_csv('EmoBankTest.tsv', sep= '\t')

