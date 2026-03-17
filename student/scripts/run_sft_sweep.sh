#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash scripts/run_sft_sweep.sh
# or override defaults, e.g.:
#   MODEL_ID=Qwen/Qwen2.5-Math-1.5B LRS="2e-5 5e-5" GLOBAL_BS="8 16" bash scripts/run_sft_sweep.sh

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B}
TRAIN_JSONL=${TRAIN_JSONL:-data/intellect_math/train}
MATH_VAL_JSONL=${MATH_VAL_JSONL:-}
MATH_TEST_JSONL=${MATH_TEST_JSONL:-}
INTELLECT_TEST_PATH=${INTELLECT_TEST_PATH:-data/intellect_math/test}

SIZES=${SIZES:-"128 256 512 1024 full"}
LRS=${LRS:-"1e-5 2e-5 5e-5"}
GLOBAL_BS=${GLOBAL_BS:-"8 16"}
MICRO_BS=${MICRO_BS:-2}
EPOCHS=${EPOCHS:-1}
EVAL_EVERY=${EVAL_EVERY:-50}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-256}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}

POLICY_DEVICE=${POLICY_DEVICE:-cuda:0}
VLLM_DEVICE=${VLLM_DEVICE:-cuda:1}

WANDB_PROJECT=${WANDB_PROJECT:-"nyu-llm-reasoners-a3-benchmarks"}
WANDB_ENTITY=${WANDB_ENTITY:-"sm12779-new-york-university"}

for size in ${SIZES}; do
  for lr in ${LRS}; do
    for gbs in ${GLOBAL_BS}; do
      run_name="sft_size${size}_lr${lr}_gbs${gbs}"
      echo "=== Running ${run_name} ==="
      uv run python student/train_sft.py \
        --model-id "${MODEL_ID}" \
        --train-jsonl "${TRAIN_JSONL}" \
        --math-val-jsonl "${MATH_VAL_JSONL}" \
        --math-test-jsonl "${MATH_TEST_JSONL}" \
        --intellect-test-path "${INTELLECT_TEST_PATH}" \
        --dataset-size "${size}" \
        --epochs "${EPOCHS}" \
        --lr "${lr}" \
        --global-batch-size "${gbs}" \
        --micro-batch-size "${MICRO_BS}" \
        --eval-every-steps "${EVAL_EVERY}" \
        --max-eval-examples "${MAX_EVAL_EXAMPLES}" \
        --max-train-steps "${MAX_TRAIN_STEPS}" \
        --policy-device "${POLICY_DEVICE}" \
        --vllm-device "${VLLM_DEVICE}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --run-name "${run_name}"
    done
  done
done

echo
echo "Sweep finished. Summarizing runs..."
uv run python student/summarize_sft_runs.py --runs-dir outputs/sft
