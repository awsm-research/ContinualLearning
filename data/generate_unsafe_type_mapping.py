import pandas as pd
import pickle

df_train = pd.read_csv("./rq1_80_10_10_split/train.csv")
df_val = pd.read_csv("./rq1_80_10_10_split/validation.csv")
df_test = pd.read_csv("./rq1_80_10_10_split/test.csv")
# Concatenate all the chunks into a single DataFrame
df = pd.concat([df_train, df_val, df_test], ignore_index=True)

categories = list(df["category"].unique())
# move 'safe' to 0 position
categories = [c for c in categories if c != 'safe']
categories.insert(0, 'safe')
category_to_index = {category: index for index, category in enumerate(categories) if str(category) != "nan" and str(category) != "SNA"}
        
with open("unsafe_type_mapping.pkl", "wb+") as f:
    pickle.dump(category_to_index, f)