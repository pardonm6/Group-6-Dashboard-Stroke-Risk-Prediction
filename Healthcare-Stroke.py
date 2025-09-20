import pandas as pd

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.to_csv("healthcare-dataset-stroke-data.csv", index=False)