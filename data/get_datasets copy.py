from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd
import json
hf_token = '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'
login(hf_token)
from datasets import load_dataset
import numpy as np



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


# Load Alpaca prompts (assumed to be safe)
with open('./alpaca_small.json') as f:
    alpaca = json.load(f)
    alpaca_prompts = [d['instruction'] for d in alpaca]


# Prepare data for DataFrame
data = []

# Add JailbreakV-28K prompts (unsafe)
data.extend([{'prompt': row['jailbreak_query'], 'category': mapping_jailbreak28k[row['policy']], 'label': 'unsafe'} for row in jailbreak28k_prompts['JailBreakV_28K']])

# Add BeaverTails prompts (unsafe)
data.extend([{'prompt': row['prompt'],'category': map_beavertail(row['category']), 'label': 'unsafe'} for row in limited_beavertail_unsafe])

# Add Alpaca prompts (safe)
data.extend([{'prompt': prompt, 'category':'safe', 'label': 'safe'} for prompt in alpaca_prompts])

# Create DataFrame
df = pd.DataFrame(data)



# Output the DataFrame
print(df.head())  # Preview the DataFrame
print(f"Total rows: {len(df)}")

# Save the DataFrame to a CSV file (optional)
df.to_csv('./prompts_with_labels_cat.csv', index=False, encoding='utf-8')


# Set random seed for reproducibility
random_seed = 42
df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle the DataFrame

# Split into 20 random chunks
num_chunks = 20
chunks = np.array_split(df, num_chunks)

# Create lists for prompts and labels from each chunk
for i, chunk in enumerate(chunks):
    prompts_list = chunk['prompt'].tolist()
    labels_list = chunk['label'].tolist()

    print(f"Chunk {i+1}:")
    print(f"Prompts: {prompts_list[:5]}")  # Print first 5 prompts for preview
    print(f"Labels: {labels_list[:5]}")   # Print first 5 labels for preview
    print(f"Chunk {i+1} contains {len(prompts_list)} items.")
    print("=" * 50)

    # Save each chunk 
    chunk.to_csv(f'./split_data_with_cat/prompts_chunk_{i+1}.csv', index=False, encoding='utf-8')