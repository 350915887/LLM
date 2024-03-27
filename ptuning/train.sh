PRE_SEQ_LEN=128
LR=2e-2
MODEL_NAME=chatglm-6b

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train  \
    --train_file ../data/train_new.json \
    --validation_file ../data/dev_new.json \
    --prompt_column question \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path  ../model/$MODEL_NAME \
    --output_dir output/$MODEL_NAME-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

