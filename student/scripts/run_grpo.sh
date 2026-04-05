#!/usr/bin/env bash
# Run a GRPO experiment on Countdown.
# Usage: bash student/scripts/run_grpo.sh [override args...]
#
# Example single run:
#   bash student/scripts/run_grpo.sh --train-steps 1000 --group-size 8
#
# Example sweep (edit the loops below):
#   LRS="1e-6 5e-6" GROUP_SIZES="4 8" bash student/scripts/run_grpo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- configurable defaults ------------------------------------------------
TRAIN_PATH="${TRAIN_PATH:-data/countdown/dataset/train}"
VAL_PATH="${VAL_PATH:-data/countdown/dataset/dev}"
PROMPT_PATH="${PROMPT_PATH:-student/prompts/countdown.prompt}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B}"
LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
N_PROMPTS="${N_PROMPTS:-1}"
GROUP_SIZES="${GROUP_SIZES:-8}"
LRS="${LRS:-1e-5}"
MICRO_BS="${MICRO_BS:-1}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MIN_GEN_TOKENS="${MIN_GEN_TOKENS:-4}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-128}"
ROLLOUT_GEN_BS="${ROLLOUT_GEN_BS:-2}"
EVAL_GEN_BS="${EVAL_GEN_BS:-4}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"
WANDB_PROJECT="${WANDB_PROJECT:-nyu-llm-reasoners-a3-grpo}"
WANDB_ENTITY="${WANDB_ENTITY:-sm12779-new-york-university}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo}"
EXTRA_ARGS="${@}"   # pass-through any extra CLI args

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for GS in ${GROUP_SIZES}; do
  for LR in ${LRS}; do
    RUN_NAME="grpo_${LOSS_TYPE}_gs${GS}_lr${LR}_steps${TRAIN_STEPS}"
    echo "======================================================"
    echo "  STARTING: ${RUN_NAME}"
    echo "======================================================"

    uv run python student/train_grpo.py \
      --train-path        "${TRAIN_PATH}" \
      --val-path          "${VAL_PATH}" \
      --prompt-path       "${PROMPT_PATH}" \
      --model-id          "${MODEL_ID}" \
      --loss-type         "${LOSS_TYPE}" \
      --train-steps       "${TRAIN_STEPS}" \
      --n-prompts-per-step "${N_PROMPTS}" \
      --group-size        "${GS}" \
      --micro-batch-size  "${MICRO_BS}" \
      --lr                "${LR}" \
      --eval-every        "${EVAL_EVERY}" \
      --eval-prompts      "${EVAL_PROMPTS}" \
      --max-seq-len       "${MAX_SEQ_LEN}" \
      --min-gen-tokens    "${MIN_GEN_TOKENS}" \
      --max-gen-tokens    "${MAX_GEN_TOKENS}" \
      --rollout-generate-batch-size "${ROLLOUT_GEN_BS}" \
      --eval-generate-batch-size    "${EVAL_GEN_BS}" \
      --vllm-gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
      --run-name          "${RUN_NAME}" \
      --output-dir        "${OUTPUT_DIR}" \
      --wandb-project     "${WANDB_PROJECT}" \
      --wandb-entity      "${WANDB_ENTITY}" \
      ${EXTRA_ARGS}

    echo "  DONE: ${RUN_NAME}"
  done
done

echo ""
echo "All runs complete. Results in ${OUTPUT_DIR}"
echo "Plot with: uv run python student/plot_grpo_results.py --runs-dir ${OUTPUT_DIR}"
