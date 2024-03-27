PRE_SEQ_LEN=128
MODEL_NAME=chatglm-6b

CUDA_VISIBLE_DEVICES=0 python web_demo.py \
    --model_name_or_path model\\$MODEL_NAME \
    --ptuning_checkpoint output/adgen-$MODEL_NAME-pt-128-2e-2/checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN

