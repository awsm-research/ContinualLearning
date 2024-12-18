import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./rq3_test_jailbreak/train.csv")
train, val = train_test_split(df, test_size=0.1, random_state=42)
train.to_csv("./rq3_test_jailbreak/train.csv")
val.to_csv("./rq3_test_jailbreak/validation.csv")

df = pd.read_csv("./rq3_test_unsafe/train.csv")
train, val = train_test_split(df, test_size=0.1, random_state=42)
train.to_csv("./rq3_test_unsafe/train.csv")
val.to_csv("./rq3_test_unsafe/validation.csv")