NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
--rlhf_type grpo \
--model_type got_ocr2 \
--ckpt_dir /home/chenzhuofan/czf/tuzhi_extract/output_sft/open2_mini/v0-20250510-185809/checkpoint-155 \
--external_plugins /home/chenzhuofan/czf/tuzhi_extract/grpo/plugin.py \
--reward_funcs value_sequence_reward2 \
--train_type full \
--torch_dtype bfloat16 \
--dataset '/home/chenzhuofan/czf/tuzhi_extract/data/grpo/open2.jsonl' \
--max_length 12800 \
--max_completion_length 128 \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--learning_rate 1e-6 \
--gradient_accumulation_steps 2 \
--save_strategy 'steps' \
--eval_strategy 'steps' \
--eval_steps 1000 \
--save_steps 1000 \
--save_total_limit 5 \
--logging_steps 1 \
--output_dir /home/chenzhuofan/czf/tuzhi_extract/output_grpo/open2 \
--warmup_ratio 0.01 \
--dataloader_num_workers 4 \
--num_generations 4 \
--temperature 1.0 \
--log_completions true \
--num_iterations 1 \
--num_infer_workers 2 \
--async_generate false \
--beta 0.001 \

