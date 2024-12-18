python main.py \
    --output_dir=./saved_models \
    --model_name=t5_large_model.bin \
    --tokenizer_name=google-t5/t5-large \
    --model_name_or_path=google-t5/t5-large \
    --train_data_file=../../data/rq1_80_10_10_split/train.csv \
    --eval_data_file=../../data/rq1_80_10_10_split/validation.csv \
    --test_data_file=../../data/rq1_80_10_10_split/test.csv \
    --safe_prompt_index 0 \
    --do_train \
    --do_test \
    --block_size 1024 \
    --epochs 5 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_t5_large.log