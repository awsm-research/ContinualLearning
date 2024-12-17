from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd
import json
hf_token = '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'
login(hf_token)
from datasets import load_dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split


## Load the JailbreakV-28K dataset and filter for unsafe prompts
jailbreak28k = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K")
jailbreak28k_prompts = jailbreak28k.filter(lambda x: x['transfer_from_llm'] == True)


# Load the BeaverTails dataset and filter for unsafe prompts
beavertail = load_dataset('PKU-Alignment/BeaverTails')
beavertail_unsafe = beavertail['330k_train'].filter(lambda x: x['is_safe'] == False)



mapping_beavertail = {
    "violence,aiding_and_abetting,incitement": "S1",
    "terrorism,organized_crime": "S1",
    "animal_abuse": "S1",
    "financial_crime,property_crime,theft": "S2",
    "drug_abuse,weapons,banned_substance": "S2",
    "non_violent_unethical_behavior": "S2",
    "child_abuse": "S4",
    "discrimination,stereotype,injustice": "S5",
    "misinformation_regarding_ethics,laws_and_safety": "S6",
    "privacy_violation": "S7",
    "hate_speech,offensive_language": "S9",
    "controversial_topics,politics": "S10",
    "self_harm": "S11",
    "sexually_explicit,adult_content": "S12"
}

['animal_abuse', 'child_abuse', 'controversial_topics,politics', 'discrimination,stereotype,injustice', 'drug_abuse,weapons,banned_substance', 'financial_crime,property_crime,theft', 'hate_speech,offensive_language', 'misinformation_regarding_ethics,laws_and_safety', 'non_violent_unethical_behavior', 'privacy_violation', 'self_harm', 'sexually_explicit,adult_content', 'terrorism,organized_crime', 'violence,aiding_and_abetting,incitement']
def map_beavertail(category):
    for key,value in category.items():
        if value == True:
            return mapping_beavertail[key]


mapping_jailbreak28k = {
    "Violence": "S1",
    "Physical Harm": "S1",
    "Animal Abuse": "S1",
    "Illegal Activity": "S2",
    "Fraud": "S2",
    "Economic Harm": "S2",
    "Child Abuse": "S4",
    "Unethical Behavior": "S6",
    "Tailored Unlicensed Advice": "S6",
    "Health Consultation": "S6",
    "Privacy Violation": "S7",
    "Bias": "S10",
    "Government Decision": "S10",
    "Hate Speech": "S10",
    "Political Sensitivity": "S10",
    "Malware": "S14"
}



# Limit beavertail_unsafe to 20,000 rows
limited_beavertail_unsafe = beavertail_unsafe.select(range(20000))  # Use the first 20,000 rows
# print(limited_beavertail_unsafe['category'][1].keys())



# Load Alpaca prompts (assumed to be safe)
with open('./alpaca_small.json') as f:
    alpaca = json.load(f)
    alpaca_prompts = [d['instruction'] for d in alpaca]
with open('./JAILJUDGE_TRAIN.json') as f:
    jailjudge = json.load(f)
    # alpaca_prompts = [d['instruction'] for d in alpaca]


# Prepare data for DataFrame
data = []

data.extend([{'prompt': row['user_prompt'], 'category': row['hazard_cate_llamaguard3'], 'label': 'unsafe', 'origin':'jailjudge'} for row in jailjudge ])


# Add JailbreakV-28K prompts (unsafe)
data.extend([{'prompt': row['jailbreak_query'], 'category': mapping_jailbreak28k[row['policy']], 'label': 'unsafe', 'origin':'jailbreak28k'} for row in jailbreak28k_prompts['JailBreakV_28K']])

# Add BeaverTails prompts (unsafe)
data.extend([{'prompt': row['prompt'],'category': map_beavertail(row['category']), 'label': 'unsafe', 'origin':'beavertail'} for row in limited_beavertail_unsafe])


# Add Alpaca prompts (safe)
data.extend([{'prompt': prompt, 'category': None, 'label': 'safe', 'origin':'alpaca'} for prompt in alpaca_prompts])



# Create DataFrame
df = pd.DataFrame(data)

# Output the DataFrame
print(df.head())  # Preview the DataFrame
print(f"Total rows: {len(df)}")


# Save the DataFrame to a CSV file (optional)
df.to_csv('./prompts_with_labels_cat.csv', index=False, encoding='utf-8')

# 1. Split by unsafe types + random selection of safe prompts

# Directory to save the CSV files
output_dir = "split_by_unsafe_type"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Filter safe and unsafe prompts
safe_prompts = df[df["label"] == "safe"]
unsafe_prompts = df[df["label"] == "unsafe"]

# Step 2: Group unsafe prompts by category
categories = unsafe_prompts["category"].unique()

unsafe_groups = {cat: unsafe_prompts[unsafe_prompts["category"] == cat] for cat in categories}

# Step 3: Distribute safe prompts equally
num_categories = len(categories)
safe_per_category = len(safe_prompts) // num_categories  # Equal share
remaining_safe = len(safe_prompts) % num_categories  # Leftover safe prompts

# Shuffle safe prompts for randomness
safe_prompts = safe_prompts.sample(frac=1, random_state=42).reset_index(drop=True)

# Allocate safe prompts to each category
safe_allocations = {}
start_idx = 0

for i, category in enumerate(categories):
    num_to_allocate = safe_per_category + (1 if i < remaining_safe else 0)
    safe_allocations[category] = safe_prompts.iloc[start_idx:start_idx + num_to_allocate]
    start_idx += num_to_allocate

# Step 4: Combine unsafe and allocated safe prompts, split into train/validation, and save
for category in categories:
    # Combine unsafe prompts and allocated safe prompts for this category
    combined_df = pd.concat([unsafe_groups[category], safe_allocations[category]], ignore_index=True)

    # Split into 90% train and 10% validation
    train_df, val_df = train_test_split(combined_df, test_size=0.1, random_state=42, shuffle=True)

    # Create a directory for this category
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    # Save train and validation splits
    train_df.to_csv(os.path.join(category_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(category_dir, "validation.csv"), index=False)

    print(f"Saved: {category_dir}/train.csv and {category_dir}/validation.csv")



# 2. Split by 80/10/10

# Step 1: Split 80% for Training and 20% for Temp (Validation + Test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Step 2: Split 20% (Temp) into 10% Validation and 10% Test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Step 3: Save the splits to CSV
train_df.to_csv("./80_10_10_split/train.csv", index=False)
val_df.to_csv("./80_10_10_split/validation.csv", index=False)
test_df.to_csv("./80_10_10_split/test.csv", index=False)

# Print Sizes
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# 3. Split into safe + unsafe, and another file of all jailbreak attacks 

alpaca_df = df[df["origin"] == "alpaca"]
alpaca_train, alpaca_test = train_test_split(alpaca_df, test_size=0.5, random_state=42, shuffle=True)



unsafe_df = df[df["origin"].isin(['beavertail'])]
jailbreak_df = df[df["origin"].isin(['jailjudge','jailbreak28k'])]

train_df = pd.concat([alpaca_train, unsafe_df], ignore_index=True)
test_df = pd.concat([alpaca_test, jailbreak_df], ignore_index=True)


output_dir = "test_jailbreak"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(f"./{output_dir}/train.csv", index=False)
test_df.to_csv(f"./{output_dir}/test.csv", index=False)

# 4. Split into safe + jailbreak, and another file of all unsafe prompts
jailbreak_df = df[df["origin"].isin(['jailjudge','jailbreak28k'])]
unsafe_df = df[df["origin"] == 'beavertail']

train_df = pd.concat([alpaca_train, jailbreak_df], ignore_index=True)
test_df = pd.concat([alpaca_test, unsafe_df], ignore_index=True)

output_dir = "test_unsafe"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(f"./{output_dir}/train.csv", index=False)
test_df.to_csv(f"./{output_dir}/test.csv", index=False)


# # Set random seed for reproducibility
# random_seed = 42
# df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle the DataFrame

# # Split into 20 random chunks
# num_chunks = 20
# chunks = np.array_split(df, num_chunks)

# # Create lists for prompts and labels from each chunk
# for i, chunk in enumerate(chunks):
#     prompts_list = chunk['prompt'].tolist()
#     labels_list = chunk['label'].tolist()

#     print(f"Chunk {i+1}:")
#     print(f"Prompts: {prompts_list[:5]}")  # Print first 5 prompts for preview
#     print(f"Labels: {labels_list[:5]}")   # Print first 5 labels for preview
#     print(f"Chunk {i+1} contains {len(prompts_list)} items.")
#     print("=" * 50)

#     # Save each chunk 
#     chunk.to_csv(f'./split_data_with_cat/prompts_chunk_{i+1}.csv', index=False, encoding='utf-8')