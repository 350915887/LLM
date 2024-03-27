PRE_SEQ_LEN=128
STEP=3000
MODEL_NAME=chatglm-6b

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_predict \
    --validation_file ../data/dev_new.json \
    --test_file ../data/dev_new.json \
    --overwrite_cache \
    --prompt_column question \
    --response_column answer \
    --model_name_or_path ../model/$MODEL_NAME \
    --ptuning_checkpoint output/$MODEL_NAME-pt-128-2e-2/checkpoint-$STEP \
    --output_dir output/$MODEL_NAME-pt-128-2e-2 \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
