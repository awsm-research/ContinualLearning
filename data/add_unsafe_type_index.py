import pandas as pd
import pickle


with open("./unsafe_type_mapping.pkl", "rb") as f:
    unsafe_type_map = pickle.load(f)


""" RQ1
split = ["train.csv", "validation.csv", "test.csv"]
for s in split:
    # ADDED
    unsafe_type_index = []
    label_index = []
    
    df = pd.read_csv(f"./rq1_80_10_10_split/{s}")
    for i in range(len(df)):
        # ADDED
        unsafe_type = df["category"][i]
        unsafe_type_index.append(unsafe_type_map[unsafe_type])
        if df["label"][i] == "safe":
            label_index.append(0)
        elif df["label"][i] == "unsafe":
            label_index.append(1)
        else:
            print("error")
            exit()
    # ADDED
    df["unsafe_type_index"] = unsafe_type_index
    df["label_index"] = label_index
    
    df.to_csv(f"./rq1_80_10_10_split/{s}", index=False)
"""

"""RQ3
folders = ["rq3_test_jailbreak", "rq3_test_unsafe"]
split = ["train.csv", "test.csv"]
for f in folders:
    for s in split:
        df = pd.read_csv(f"./{f}/{s}")
        
        # ADDED
        unsafe_type_index = []
        label_index = []
        
        for i in range(len(df)):
            # ADDED
            unsafe_type = df["category"][i]
            unsafe_type_index.append(unsafe_type_map[unsafe_type])
            if df["label"][i] == "safe":
                label_index.append(0)
            elif df["label"][i] == "unsafe":
                label_index.append(1)
            else:
                print("error")
                exit()
        # ADDED
        df["unsafe_type_index"] = unsafe_type_index
        df["label_index"] = label_index
        
        
        df.to_csv(f"./{f}/{s}", index=False)
"""

"""RQ2
split = ["train.csv", "validation.csv"]

for x in range(1, 15, 1):
    folder_name = f"S{x}"
    for s in split:
        df = pd.read_csv(f"./rq2_split_by_unsafe_type/{folder_name}/{s}")
        
        # ADDED
        unsafe_type_index = []
        label_index = []
        
        for i in range(len(df)):
            # ADDED
            unsafe_type = df["category"][i]
            unsafe_type_index.append(unsafe_type_map[unsafe_type])
            if df["label"][i] == "safe":
                label_index.append(0)
            elif df["label"][i] == "unsafe":
                label_index.append(1)
            else:
                print("error")
                exit()
        # ADDED
        df["unsafe_type_index"] = unsafe_type_index
        df["label_index"] = label_index
                
        df.to_csv(f"./rq2_split_by_unsafe_type/{folder_name}/{s}", index=False)
"""