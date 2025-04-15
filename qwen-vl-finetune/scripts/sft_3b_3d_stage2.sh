#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
NPROC_PER_NODE=8

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=./Qwen2.5-VL-3B-Instruct-SFT3D-coco3d-scannet2d-stage1


# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=coco_complex_reasoning_3d_77k,coco_conversation_3d_58k,coco_detail_3d_23k,coco_3dcoord_grounding,scannet_2d_embodied_dialogue_train,scannet_2d_embodied_planning_train,scannet_2d_embodied_qa_train,scannet_2d_room_description_train,scannet_2d_3dcoord_grounding_train

# Output configuration
run_name="qwen2.5-3b-3dvl-coco3d-scannet2d-stage2"
output_dir=./Qwen2.5-VL-3B-Instruct-SFT3D-coco3d-scannet2d-stage2

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --enable_3d \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --tune_mm_coord True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate ${lr} \
    --mm_projector_lr 2e-6 \
    --vision_tower_lr ${lr} \
    --coord_tower_lr 2e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}