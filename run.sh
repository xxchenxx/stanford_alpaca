NCCL_P2P_DISABLE=1 nohup torchrun --nproc_per_node=4 --master_port=29510 train.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir full_training \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config fsdp.json \
    --tf32 True &




NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --master_port=29510 train.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir full_training \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config fsdp.json \
    --tf32 True \
    --report_to none



    


NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --master_port=29510 train.py \
    --model_name_or_path facebook/opt-125m \
    --data_path ./alpaca_data.json \
    --bf16 False \
    --model_max_length 128 \
    --output_dir full_training \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config fsdp_default.json \
    --tf32 True \
    --report_to none \
    --dataloader_pin_memory False


NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --master_port=29510 train_no_callback.py \
    --model_name_or_path facebook/opt-125m \
    --data_path ./alpaca_data.json \
    --bf16 False \
    --model_max_length 128 \
    --output_dir full_training_no_callback \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config fsdp_default.json \
    --tf32 True \
    --report_to none \
    --dataloader_pin_memory False