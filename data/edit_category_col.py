import pandas as pd

"""RQ1
split = ["train.csv", "validation.csv", "test.csv"]

for s in split:
    df = pd.read_csv(f"./rq1_80_10_10_split/{s}")
    # drop SNA
    df = df[df["category"] != "SNA"].reset_index(drop=True)
    for i in range(len(df)):
        if str(df["category"][i]) == "nan":
            df.loc[i, "category"] = "safe"
    print(df["category"].unique())
    df.to_csv(f"./rq1_80_10_10_split/{s}", index=False)
"""

"""RQ3
folders = ["rq3_test_jailbreak", "rq3_test_unsafe"]
split = ["train.csv", "test.csv"]
for f in folders:
    for s in split:
        df = pd.read_csv(f"./{f}/{s}")
        # drop SNA
        df = df[df["category"] != "SNA"].reset_index(drop=True)
        for i in range(len(df)):
            if str(df["category"][i]) == "nan":
                df.loc[i, "category"] = "safe"
        print(df["category"].unique())
        df.to_csv(f"./{f}/{s}", index=False)
"""

"""RQ2
split = ["train.csv", "validation.csv"]


for x in range(1, 15, 1):
    folder_name = f"S{x}"
    for s in split:
        df = pd.read_csv(f"./rq2_split_by_unsafe_type/{folder_name}/{s}")
        # drop SNA
        df = df[df["category"] != "SNA"].reset_index(drop=True)
        for i in range(len(df)):
            if str(df["category"][i]) == "nan":
                df.loc[i, "category"] = "safe"
        print(df["category"].unique())
        df.to_csv(f"./rq2_split_by_unsafe_type/{folder_name}/{s}", index=False)
"""