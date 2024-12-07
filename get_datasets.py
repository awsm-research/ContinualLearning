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

# Load Alpaca prompts (assumed to be safe)
with open('./alpaca_small.json') as f:
    alpaca = json.load(f)
    alpaca_prompts = [d['instruction'] for d in alpaca]

# Prepare data for DataFrame
data = []

# Add JailbreakV-28K prompts (unsafe)
data.extend([{'prompt': row['jailbreak_query'], 'label': 'unsafe'} for row in jailbreak28k_prompts['JailBreakV_28K']])

# Add BeaverTails prompts (unsafe)
data.extend([{'prompt': row['prompt'], 'label': 'unsafe'} for row in beavertail_unsafe])

# Add Alpaca prompts (safe)
data.extend([{'prompt': prompt, 'label': 'safe'} for prompt in alpaca_prompts])

# Create DataFrame
df = pd.DataFrame(data)

# Output the DataFrame
print(df.head())  # Preview the DataFrame
print(f"Total rows: {len(df)}")

# Save the DataFrame to a CSV file (optional)
df.to_csv('./prompts_with_labels.csv', index=False)


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
    chunk.to_csv(f'./split_data/prompts_chunk_{i+1}.csv', index=False)