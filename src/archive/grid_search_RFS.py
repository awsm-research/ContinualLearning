import os
import itertools

# Define hyperparameter ranges
learning_rates = [5e-4, 1e-4, 5e-5, 1e-5]
lora_rs = [8, 16, 32]
lora_alphas = [16, 32, 64]
lora_dropouts = [0.1, 0.2, 0.3]

# Fixed parameters
base_train_cmd = (
    "python RFS_train_cat.py --do_train --batch_size=1 --data_dir=split_data_with_cat "
    "--model_name_or_path=meta-llama/Llama-Guard-3-1B "
    "--saved_model_name={saved_model_name} --epochs=1 --max_grad_norm=1.0 "
    "--max_train_input_length=2048 --chunks=1 --max_new_tokens=6 --seed 123456 "
    "--learning_rate={lr} --lora_r={lora_r} --lora_alpha={lora_alpha} --lora_dropout={lora_dropout} "
    "2>&1 | tee {log_file}"
)

base_test_cmd = (
    "python RFS_train_cat.py --do_test --batch_size=1 --data_dir=split_data_with_cat "
    "--model_name_or_path=meta-llama/Llama-Guard-3-1B "
    "--saved_model_name={saved_model_name} --chunks=2 "
    "--test_result_file={test_result_file} --max_new_tokens=6 --seed 123456 "
    "2>&1 | tee {test_log_file}"
)

# Iterate over all combinations
combinations = itertools.product(learning_rates, lora_rs, lora_alphas, lora_dropouts)

for i, (lr, lora_r, lora_alpha, lora_dropout) in enumerate(combinations):
    # Define unique identifiers for the model and logs
    saved_model_name = f"RFS_cat_chunk1_lr{lr}_r{lora_r}_alpha{lora_alpha}_dropout{str(lora_dropout).strip('.')}"
    log_file = f"{saved_model_name}.log"
    test_result_file = f"{saved_model_name}_test_result"
    test_log_file  = f"{saved_model_name}_test.log"

    # Construct the commands
    train_cmd = base_train_cmd.format(
        saved_model_name=saved_model_name,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        log_file=log_file
    )

    test_cmd = base_test_cmd.format(
        saved_model_name=saved_model_name,
        test_result_file=test_result_file,
        test_log_file = test_log_file
    )

    # Run the training command
    print(f"Running training for combination {i+1}: lr={lr}, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    os.system(train_cmd)

    # Run the testing command
    print(f"Running testing for combination {i+1}: {test_result_file}")
    os.system(test_cmd)
