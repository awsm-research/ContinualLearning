import pandas as pd

split = ["train.csv", "validation.csv", "test.csv"]

for s in split:
    df = pd.read_csv(f"./rq1_80_10_10_split/{s}")    
    df = df.dropna(subset=['prompt']).reset_index(drop=True)
    df.to_csv(f"./rq1_80_10_10_split/{s}", index=False)

folders = ["rq3_test_jailbreak", "rq3_test_unsafe"]
split = ["train.csv", "test.csv"]
for f in folders:
    for s in split:
        df = pd.read_csv(f"./{f}/{s}")
        df = df.dropna(subset=['prompt']).reset_index(drop=True)
        df.to_csv(f"./{f}/{s}", index=False)

split = ["train.csv", "validation.csv"]
for x in range(1, 15, 1):
    folder_name = f"S{x}"
    for s in split:
        df = pd.read_csv(f"./rq2_split_by_unsafe_type/{folder_name}/{s}")
        df = df.dropna(subset=['prompt']).reset_index(drop=True)
        df.to_csv(f"./rq2_split_by_unsafe_type/{folder_name}/{s}", index=False)