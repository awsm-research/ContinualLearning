import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import DisjunctiveConstraint
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict
from tqdm import tqdm
import os
from transformers import get_scheduler
import argparse
import logging
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import re
from torch.nn import CrossEntropyLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123456,
                    help="random seed for initialization")

parser.add_argument("--data_dir", default="data", type=str,
                    help="Folder name for the data.")
parser.add_argument("--do_train", action='store_true', default=False,
                    help="Whether to train the model.")
parser.add_argument("--do_test", action='store_true', default=False,
                    help="Whether to run inference.")
parser.add_argument("--do_ablation", action='store_true', default=False,
                    help="Whether to do ablation study on the amount of training data.")
# parser.add_argument("--training_proportion", default=None, type=int,
#                     help="")

parser.add_argument("--model_name_or_path", default="meta-llama/Llama-Guard-3-1B", type=str,
                    help="The model checkpoint for weights initialization.")
parser.add_argument("--saved_model_name", default=None, type=str,
                    help="Name for the best saved model.")
parser.add_argument("--test_result_file", default=None, type=str,
                    help="File name and path for writing testing results.")

parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="LoRA parameters")
parser.add_argument("--epochs", default=3, type=int,
                    help="LoRA parameters")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Maximum number of subword tokens per input.")

parser.add_argument("--lora_r", default=8, type=int,
                    help="LoRA parameters")
parser.add_argument("--lora_alpha", default=32, type=int,
                    help="LoRA parameters")
parser.add_argument("--lora_dropout", default=0.1, type=float,
                    help="LoRA parameters")

parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for train/val/test")
parser.add_argument("--max_train_input_length", default=2048, type=int,
                    help="Maximum number of subword tokens per input.")
parser.add_argument("--max_new_tokens", default=5, type=int,
                    help="Generation parameters")

parser.add_argument("--chunks", default=1, type=int,
                    help="Number of chunks used in training.")                 

args = parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

set_seed(args)

# Create a PyTorch Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

def check_model_output(output):
    """
    Checks if the model output follows the expected format.

    Args:
        output (str): The model output.

    Returns:
        bool: True if the output format is correct, False otherwise.
    """
    output = output.strip()
    
    # Check if the output is exactly 'safe'
    if output == "safe":
        return True
    
    # Check if the output matches the pattern 'unsafe' followed by a newline and 'S{number}' (1-14)
    pattern = r"^unsafe\nS(1[0-4]|[1-9])$"
    if re.match(pattern, output):
        return True
    
    # If neither condition is satisfied, return False
    return False

def tokenize_with_template(user_inputs, agent_labels,categories):
    input_ids_list = []
    labels_list = []
    
    for user_input, agent_label, category in tqdm(zip(user_inputs, agent_labels, categories), desc="Tokenizing"):
        # Apply the training chat template
        chat = [
            {"role": "user", "content": [
            {
                "type": "text", 
                "text": user_input
            },
        ],
        },
        ]

        # Format the chat into a single string (without tokenizing yet)
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", max_length=args.max_train_input_length
                                                  , padding="max_length", truncation=True).squeeze(0)
        
        # Combine label and category with newline
        if agent_label == 'safe':
            target_text = agent_label 
            label_input = tokenizer(target_text, padding="max_length", truncation=True, return_tensors="pt", max_length=3)["input_ids"].squeeze(0)
        else:
            target_text = f"\n\n{agent_label}\n{category}"
            label_input = tokenizer(target_text, padding="max_length", truncation=True, return_tensors="pt", max_length=8)["input_ids"].squeeze(0)
        

        # Tokenize the combined label and category
        label_input = tokenizer(target_text, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)

        # Prepare the labels tensor with -100 (ignore index for loss calculation)
        labels = torch.full_like(input_ids, -100)

        # Insert the label sequence at the appropriate position
        first_pad_index = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]  # First pad index
        labels[first_pad_index:first_pad_index + label_input.size(0)] = label_input  # Place the label sequence
                
        # print(first_pad_index)
        # print(labels)
        # exit()
        
        # Append the input_ids and labels to their respective lists
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        
    # Stack all tokenized inputs and labels into single tensors
    encodings = {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list)
    }
    
    return encodings

def tokenize_with_template_test(user_inputs, agent_labels, categories):
    input_ids_list = []
    labels_list = []

    for user_input, agent_label, category in tqdm(zip(user_inputs, agent_labels, categories), desc="Tokenizing"):
        # Apply the training chat template
        chat = [
            {"role": "user", "content": [
            {
                "type": "text", 
                "text": user_input
            },
        ],
        },
        ]
        # Format the chat into a single string (without tokenizing yet)
        input_ids = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            max_length=args.max_train_input_length,
            truncation=True, 
            padding="max_length" # Remove `padding` to avoid adding extra tokens
        ).squeeze(0)

        # Combine label and category with newline
        
        if agent_label == 'safe':
            target_text = agent_label 
            label_input = tokenizer(target_text, padding="max_length", truncation=True, return_tensors="pt", max_length=3)["input_ids"].squeeze(0)
        else:
            target_text = f"\n\n{agent_label}\n{category}"
            label_input = tokenizer(target_text, padding="max_length", truncation=True, return_tensors="pt", max_length=7)["input_ids"].squeeze(0)
        

        # Tokenize the combined label and category
        # label_input = tokenizer(target_text, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
        
        
        # print(label_input)
        # print(tokenizer.decode(input_ids, skip_special_tokens=False))
        # exit()
        # Prepare the labels tensor with -100 (ignore index for loss calculation)
        labels = torch.full_like(input_ids, -100)

        # Insert the label sequence at the appropriate position
        first_pad_index = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]  # First pad index

        labels[first_pad_index:first_pad_index + label_input.size(0)] = label_input  # Place the label sequence
        # print(first_pad_index)
        # print(labels[first_pad_index:first_pad_index + label_input.size(0)])
        # exit()

        
        # Append the input_ids and labels to their respective lists
        input_ids_list.append(input_ids)
        labels_list.append(labels)


    encodings = {
        "input_ids": input_ids_list,
        "labels": labels_list
    }
    return encodings

if args.do_train:


    # List to store data from multiple chunks
    train_dfs = []

    # Iterate through the specified number of chunks
    for i in range(1, args.chunks + 1):  # From 1 to args.chunk (inclusive)
        print(f'Loading chunk {i}')
        chunk_path = f"./{args.data_dir}/prompts_chunk_{i}.csv"
        chunk_df = pd.read_csv(chunk_path)  # Load each chunk
        train_dfs.append(chunk_df)  # Add the DataFrame to the list

    # Concatenate all the chunks into a single DataFrame
    train_df = pd.concat(train_dfs, ignore_index=True)
    # Shuffle the combined DataFrame
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Extract prompts, labels, and categories
    train_texts = train_df["prompt"].tolist()
    train_labels = train_df["label"].tolist()
    train_categories = train_df["category"].tolist()


# # Tokenize with label and category
# train_encodings = tokenize_with_template(train_texts, train_labels, train_categories)

if args.do_test:

    chunk_path = f"./{args.data_dir}/prompts_chunk_{args.chunks}.csv"
    test_df = pd.read_csv(chunk_path)
    test_df = test_df.head(100)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_texts = test_df["prompt"].tolist()
    test_labels = test_df["label"].tolist()
    test_categories = test_df["category"].tolist()

# load tokenizer and model
if args.do_train:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # meta-llama/Llama-Guard-3-1B
elif args.do_test:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = "<|finetune_right_pad_id|>"
#     tokenizer.add_special_tokens({'pad_token': "<|finetune_right_pad_id|>"})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.pad_token = "<pad>"
    tokenizer.add_special_tokens({'pad_token': "<pad>"})



if args.do_train:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                use_cache=False)
    model.resize_token_embeddings(len(tokenizer))
    # Apply LoRA configuration
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"] 
    ### Default: trainable params: 4,718,592 || all params: 8,034,979,840 || trainable%: 0.0587 ###
    lora_config = LoraConfig(task_type="CAUSAL_LM",
                            inference_mode=False,
                            r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout,
                            target_modules=target_modules,
                            bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    

# Create DataLoader for batching
if args.do_train:
    train_encodings = tokenize_with_template(train_texts, train_labels,train_categories)
    train_dataset = TextDataset(train_encodings)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

if args.do_test:
    test_encodings = tokenize_with_template_test(test_texts, test_labels,test_categories)
    test_dataset = TextDataset(test_encodings)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)


regex_pattern = r"^(safe|\\n\\nunsafe\\nS(1[0-4]|[1-9]))$"

def regex_penalty_loss(decoded_outputs, penalty_weight=1):
    penalty_loss = 0.0
    for output in decoded_outputs:
        if not re.match(regex_pattern, output):
            penalty_loss += penalty_weight  # Add penalty for invalid output
    return penalty_loss / len(decoded_outputs)  # Normalize by batch size


# start training
if args.do_train:
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    num_train_steps = len(train_dataloader) * args.epochs  # Number of batches * epochs
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    lr_scheduler = get_scheduler(
                                    name="constant",
                                    optimizer=optimizer,
                                    num_warmup_steps=0,  # No warmup for constant scheduler
                                    num_training_steps=num_train_steps  # Total training steps
                                )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader) * args.batch_size)
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Train batch size = %d", args.batch_size)
    logger.info("  Test batch size = %d", args.batch_size)
    logger.info("  Gradient Clipping = %d", args.max_grad_norm)
    logger.info("  Total optimization steps = %d", num_train_steps)
    device = "cuda:0"
    # Training loop
    best_loss = 1e6
    model.train()
    for epoch in range(args.epochs):  # Number of epochs
        total_loss = 0
        tr_loss = 0
        tr_num = 0
        avg_loss = 0
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        for batch in bar:
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            # base_loss = outputs.loss  # Original loss from the model
        
            # # Decode outputs to strings
            logits = outputs.logits
            # decoded_outputs = tokenizer.batch_decode(
            #     torch.argmax(logits, dim=-1), skip_special_tokens=False
            # )
            # print(decoded_outputs)

            # # Compute regex penalty loss
            # penalty_weight = 0.5  # Adjust the weight of the penalty
            # penalty_loss = regex_penalty_loss(decoded_outputs, penalty_weight=penalty_weight)
            
            # # Combine base loss and penalty loss
            # loss = base_loss + penalty_loss
            loss = loss_fn(outputs.logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()  # Update the learning rate

            tr_loss = loss.item()
            total_loss += tr_loss
            tr_num += 1
            avg_loss = round(total_loss/tr_num, 5)
            
            bar.set_description("epoch {} loss {}".format(epoch, avg_loss))

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss}")
        
        # Save the model
        best_model_path = f"./{args.saved_model_name}"
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        logger.info("Saving model checkpoint to %s", best_model_path)
        
        model.train()

if args.do_test:
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataloader)*args.batch_size)
    logger.info("  Batch size = %d", args.batch_size)
    
    # 1 GPU for testing
    device = "cuda:0"
    valid_outputs = ["safe"] + [f"\n\nunsafe\nS{i}" for i in range(1, 15)]
    # Tokenize valid outputs for constraints
    valid_output_ids = [tokenizer(output, add_special_tokens=False)["input_ids"] for output in valid_outputs]

    # Create DisjunctiveConstraint
    constraints = [DisjunctiveConstraint(valid_output_ids)]

    # load base model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-1B",
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                use_cache=False)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model,
                                    args.saved_model_name,
                                    device_map=device,
                                    torch_dtype=torch.bfloat16,
                                    use_cache=False)
    
    model = model.merge_and_unload()

    # Merge LoRA weights into the base model
    model.eval()
    
    # for valid_output in valid_outputs:
    #     print(f"Output: {valid_output}, Tokens: {tokenizer(valid_output, add_special_tokens=False)['input_ids']}")
    # exit()
    label_predictions = []
    label_targets = []
    type_predictions = []
    type_targets = []
    type_correctness = []
    # Loop through the test dataloader
    count = 0
    correct_count = 0
    pred_err_count = 0
    model_output_err_count = 0
    predictions = []

    with torch.no_grad():
        test_bar = tqdm(test_dataloader, total=len(test_dataloader))
        for batch in test_bar:
            input_ids = batch["input_ids"].to(device) # use one GPU for testing (cuda:0)
            labels = batch["labels"].to(device)
            
            # Generate outputs using the model

            try:

                generated_outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    min_length=2,
                    # constraints=constraints,
                    # num_beams=5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.7,     # Add diversity
                    repetition_penalty=1.2,  # Penalize repeating EOS
                    do_sample=True, 
                )

            except Exception as e:
                print("Error during generation:", e)
                predictions.append(-1)
                continue
            
            # Decode and print each generated output
            for input_id, generated_output, test_label in zip(input_ids, generated_outputs, labels):
                prompt_len = input_ids.shape[-1]
                generated_output_len = generated_output.shape[-1]
                # Decode the input and output
                input_text = tokenizer.decode(input_id, skip_special_tokens=False)
                output_text = tokenizer.decode(generated_output[prompt_len:], skip_special_tokens=False)
                first_pad_index = (input_id == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0] 
                encoded_label = test_label[first_pad_index:]
                decoded_label = tokenizer.decode(encoded_label[encoded_label != -100], skip_special_tokens=False)
                decoded_output = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
                # print(f"Decoded Output: {decoded_output}")
                # exit()
                # Extract unsafe/safe label and type from the output
                if len(decoded_label) >= 1:
                    if 'unsafe' in decoded_label:
                        true_label = 'unsafe'
                        true_type = decoded_label[8:] 
                    else:
                        true_label = 'safe'
                        true_type = None
                          # Second token: unsafe type
                else:
                    continue
                print('this is decoded label: ' + decoded_label)
                print('this is output text: ' + output_text)
                
                output_text = output_text.strip()

                # Split by newlines
                parts = output_text.split("\n")
                # print(f'output len = {len(parts)}')
                
                if len(parts) >= 2:
                    pred_label = parts[0]
                    pred_type = parts[1]
                else:
                    pred_label = parts[0]
                    pred_type = None
                count += 1
                
                # Record binary metrics for unsafe/safe label
                # label_predictions.append(pred_label.strip())
                # label_targets.append(true_label.strip())

                if pred_label.strip() == "safe":
                    pred_label = 0
                elif pred_label.strip() == "unsafe":
                    pred_label = 1
                else:
                    # print("prediction error!")
                    pred_err_count += 1
                    pred_label = -1
                    
                    
                if true_label.strip() == "safe":
                    true_label = 0
                elif true_label.strip() == "unsafe":
                    true_label = 1
                else:
                    print("label error!")
                    exit()
                    
                # Record binary metrics for unsafe/safe label
                label_predictions.append(pred_label)
                label_targets.append(true_label)

                if pred_label == true_label:
                    correct_count += 1
                # # Record multi-class metrics for unsafe types
                # Record multi-class metrics for unsafe types
                if (true_label == 1) and (pred_label == 1) and true_type and pred_type:
                    type_predictions.append(pred_type.strip())
                    type_targets.append(true_type.strip())
                    type_correctness.append(pred_type.strip()==true_type.strip())
                
                predictions.append(output_text)
                
                acc = round(correct_count/count, 2)
                type_acc = 0
                if len(type_correctness)>0:
                    type_acc = sum(type_correctness)/len(type_correctness)
                
                test_bar.set_description(f"Binary Acc: {acc} | {correct_count}/{count} || Multiclass Acc: {type_acc} | {sum(type_correctness)}/{len(type_correctness)}")
    
                # if count == 20:
                #     exit()
    print(len(label_predictions))
    print(len(label_targets))
    print("Model Output Error Count: ", model_output_err_count)
    print("Pred Error Count: ", pred_err_count)
    
    # Calculate binary metrics for unsafe/safe detection
    binary_accuracy = accuracy_score(label_targets, label_predictions)

    removed_error_label, removed_error_prediction = zip(*[(x, y) for x, y in zip(label_targets, label_predictions) if y != -1])

    binary_precision = precision_score(removed_error_label, removed_error_prediction, pos_label=1)
    binary_recall = recall_score(removed_error_label, removed_error_prediction, pos_label=1)
    binary_f1 = f1_score(removed_error_label, removed_error_prediction, pos_label=1)
    
    logger.info("***** Binary Metrics for Unsafe/Safe Detection *****")
    logger.info(f"Accuracy: {binary_accuracy:.4f}")
    logger.info(f"Precision: {binary_precision:.4f}")
    logger.info(f"Recall: {binary_recall:.4f}")
    logger.info(f"F1 Score: {binary_f1:.4f}")

    # Calculate multi-class accuracy for unsafe types
    if type_targets and type_predictions:
        multi_class_accuracy = accuracy_score(type_targets, type_predictions)
        logger.info("***** Multi-Class Accuracy for Unsafe Types *****")
        logger.info(f"Accuracy: {multi_class_accuracy:.4f}")
    else:
        logger.info("No unsafe type predictions available for multi-class accuracy.")
    
    
    logger.info(f"***** Writing Testing Results to {args.saved_model_name + '/' + args.test_result_file} *****")
    test_df["label_predictions"] = label_predictions
    test_df["label_targets"] = label_targets
    test_df["type_predictions"] = ''
    test_df["type_targets"] = ''
    # test_df.to_csv(args.saved_model_name + '/' + args.test_result_file + '.csv')
    logger.info("Done.")