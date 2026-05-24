#!/usr/bin/env bash

# Credentials
export HF_TOKEN=""  # set your HuggingFace token
export HF_HOME="./data/huggingface_cache"

# Paths (set before running)
MODEL_PATH="/path/to/trained/model"   # output model directory from careR1.sh
DATA_DIR="/path/to/data"              # root data directory containing OmniMedVQA/
OUTPUT_DIR="/path/to/eval/output"     # where to save evaluation results
EVAL_SCRIPT="/path/to/eval/original_test_qwen2vl_med.py"

# Eval config
export CUDA_VISIBLE_DEVICES=0
MODALITY="XRAY"
BSZ=4
PROMPT_TYPE="complex"
MAX_NEW_TOKENS=512
DO_SAMPLE=true
TEMPERATURE=1.0
SEED=42

BASE_PATH="${DATA_DIR}/OmniMedVQA/OmniMedVQA"
PROMPT_PATH="${DATA_DIR}/OmniMedVQA/modality/test/${MODALITY}_test.json"
OUTPUT_PATH="${OUTPUT_DIR}/results_${MODALITY}.json"
LOG_PATH="${OUTPUT_DIR}/eval_${MODALITY}.log"

mkdir -p "$OUTPUT_DIR"

echo "Running: modality=${MODALITY}, bsz=${BSZ}, model=$(basename ${MODEL_PATH})"

python3 "$EVAL_SCRIPT" \
  --modality "$MODALITY" \
  --prompt_type "$PROMPT_TYPE" \
  --model_path "$MODEL_PATH" \
  --bsz "$BSZ" \
  --output_path "$OUTPUT_PATH" \
  --prompt_path "$PROMPT_PATH" \
  --base_path "$BASE_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --do_sample "$DO_SAMPLE" \
  --temperature "$TEMPERATURE" \
  ${SEED:+--seed "$SEED"} \
  >"$LOG_PATH" 2>&1

if [ $? -eq 0 ]; then
  echo "[SUCCESS] modality=${MODALITY} — see ${OUTPUT_PATH}"
else
  echo "[FAILED]  modality=${MODALITY} — see ${LOG_PATH}"
fi
