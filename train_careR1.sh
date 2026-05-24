#!/bin/bash
set -euo pipefail
cd src/open-r1-multimodal

# Credentials 
export WANDB_API_KEY=""   # set your WandB API key
export HF_TOKEN=""        # set your HuggingFace token

# Paths (set before running)
DATA_DIR="/path/to/data"      # root data directory containing Huatuo/
OUTPUT_DIR="/path/to/output"  # where to save the trained model

SCRIPT="src/open_r1/care.py"

# Training config
export GPU_NUM=""           # set gpu 
export DATASET="Huatuo"
export DATASET_SELECTION="XRAY"   # choice: XRAY, MR, Ultrasound, CT, Fundus, OCT, Dermoscopy, Microscopy
export DEBUG_MODE="true"
export HF_HOME="./data/huggingface_cache"

CORR_TYPE="black"
CORR_FOLDER="${DATA_DIR}/${DATASET}/images_corr_${DATASET_SELECTION}"

# WandB 
export WANDB_ENTITY=""    # set your WandB entity
export WANDB_PROJECT="care-r1-${DATASET_SELECTION}"
export WANDB_JOB_TYPE="training"
export WANDB_RUN_NAME="care-r1_${DATASET_SELECTION}_run-$(date +%m-%d-%H-%M)"

# Logging 
export LOG_PATH="${OUTPUT_DIR}/logs/train_log.txt"
mkdir -p "${OUTPUT_DIR}/logs"


torchrun --nproc_per_node="${GPU_NUM}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="29509" \
    "$SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "${DATA_DIR}/${DATASET}/" \
    --dataset_selection "$DATASET_SELECTION" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --max_pixels 100352 \
    --num_train_epochs 2 \
    --run_name "$WANDB_RUN_NAME" \
    --save_only_model true \
    --save_strategy "no" \
    --num_generations 8 \
    --corr_type "$CORR_TYPE" \
    --corr_folder "$CORR_FOLDER" \
    --dataloader_num_workers 8 \
2>&1 | tee "$LOG_PATH"

echo "$WANDB_RUN_NAME"
