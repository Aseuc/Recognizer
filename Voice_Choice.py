import pandas as pd


data = pd.read_excel("TDNN2Z.xlsx")
df = pd.DataFrame(data)

df_sample = df.sample(frac=1)
df_sample.to_excel("TDNN3Z.xlsx")
