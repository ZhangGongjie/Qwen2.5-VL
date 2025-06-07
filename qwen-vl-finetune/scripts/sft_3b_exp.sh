#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
NPROC_PER_NODE=8
WANDB_MODE="offline"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID


# Training hyperparameters
lr=1e-5
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=coco_complex_reasoning_77k,coco_conversation_58k,coco_detail_23k,omni3d_nuscenes_train_3d_object_detection_under_cam_coordsys%200,omni3d_nuscenes_val_3d_object_detection_under_cam_coordsys%200,omni3d_kitti_train_3d_object_detection_under_cam_coordsys%300,omni3d_kitti_val_3d_object_detection_under_cam_coordsys%300,omni3d_sunrgbd_train_3d_object_detection_under_cam_coordsys%200,omni3d_sunrgbd_val_3d_object_detection_under_cam_coordsys%200,omni3d_arkitscenes_train_3d_object_detection_under_cam_coordsys,omni3d_arkitscenes_val_3d_object_detection_under_cam_coordsys,omni3d_objectron_train_3d_object_detection_under_cam_coordsys,omni3d_objectron_val_3d_object_detection_under_cam_coordsys,omni3d_hypersim_train_3d_object_detection_under_cam_coordsys,omni3d_hypersim_val_3d_object_detection_under_cam_coordsys

# Output configuration
run_name="qwen2.5-3b-3dvl-3dod-e2e-plainqwen-camraype"
output_dir=./Qwen2.5-VL-3B-Instruct-3dvl-3dod-e2e-plainqwen-camraype

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 100352 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 20 \
    --learning_rate ${lr} \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}