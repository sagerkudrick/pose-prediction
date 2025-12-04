import pandas as pd
df = pd.read_csv("dataset_csv/rotations_20251203_150653.csv")
print(df[['x','y','z','w']].apply(lambda row: tuple(row), axis=1).value_counts())