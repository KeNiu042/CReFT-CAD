MASTER_PORT=29501 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 swift sft \
--model_type got_ocr2 \
--model /home/chenzhuofan/czf/tuzhi_extract/output_grpo/judge/v0-20250508-150838/checkpoint-309 \
--train_type full \
--num_train_epochs 3 \
--torch_dtype bfloat16 \
--dataset /home/chenzhuofan/czf/tuzhi_extract/data/sft/open2.jsonl \
--output_dir /home/chenzhuofan/czf/tuzhi_extract/output_sft/judge_open2 \
--learning_rate 2e-5 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 8 \
--eval_steps 9000 \
--save_steps 9000 \
--freeze_aligner False \
--freeze_vit False \
--ddp_find_unused_parameters true \
--gradient_checkpointing false

