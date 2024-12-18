import pandas as pd

df = pd.read_csv("./validation.csv")

for i in range(len(df)):
    if df["label_index"][i] != 0 and df["label_index"][i] != 1:
        print("error")
