from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from student.drgrpo_grader import question_only_reward_fn
from student.sft import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SFTRow:
    prompt: str
    response: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_prompt_template(prompt_path: str | Path) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


def _extract_prompt_from_example(ex: dict[str, Any]) -> str:
    if "prompt" in ex and isinstance(ex["prompt"], str):
        return ex["prompt"]

    msgs = ex.get("messages", [])
    if msgs:
        sys_msg = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        if sys_msg:
            return sys_msg + "\n\n" + user_msg
        return user_msg

    return ""


def _extract_response_from_example(ex: dict[str, Any]) -> str:
    if "response" in ex and isinstance(ex["response"], str):
        return ex["response"]

    msgs = ex.get("messages", [])
    if msgs:
        assistant_msg = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        if assistant_msg:
            return assistant_msg

    return ex.get("ground_truth", ex.get("answer", ""))


def load_sft_train_rows(path: str | Path) -> list[SFTRow]:
    p = Path(path)
    if p.is_dir():
        ds = load_from_disk(str(p))
        return [
            SFTRow(
                prompt=_extract_prompt_from_example(ex),
                response=_extract_response_from_example(ex),
            )
            for ex in ds
        ]

    rows = read_jsonl(p)
    return [
        SFTRow(
            prompt=_extract_prompt_from_example(r),
            response=_extract_response_from_example(r),
        )
        for r in rows
    ]


def load_intellect_split(path: str | None) -> tuple[list[str], list[str]]:
    if path is None:
        return [], []
    p = Path(path)
    if p.is_dir():
        ds = load_from_disk(str(p))
        prompts, gts = [], []
        for ex in ds:
            prompts.append(_extract_prompt_from_example(ex))
            gts.append(ex.get("ground_truth", ""))
        return prompts, gts

    rows = read_jsonl(p)
    prompts, gts = [], []
    for ex in rows:
        prompts.append(ex.get("prompt", ""))
        gts.append(ex.get("ground_truth", ex.get("answer", "")))
    return prompts, gts


def load_math_split(path: str | None, prompt_template: str, split: str) -> tuple[list[str], list[str]]:
    if path:
        p = Path(path)
        if p.is_dir():
            ds = load_from_disk(str(p))
            prompts, gts = [], []
            for ex in ds:
                if "problem" in ex:
                    prompts.append(prompt_template + "\n\n" + ex["problem"])
                    gts.append(ex["answer"])
                else:
                    prompts.append(_extract_prompt_from_example(ex))
                    gts.append(ex.get("ground_truth", ex.get("answer", "")))
            return prompts, gts

        rows = read_jsonl(p)
        prompts, gts = [], []
        for ex in rows:
            if "problem" in ex:
                prompts.append(prompt_template + "\n\n" + ex["problem"])
                gts.append(ex["answer"])
            else:
                prompts.append(ex.get("prompt", ""))
                gts.append(ex.get("ground_truth", ex.get("answer", "")))
        return prompts, gts

    ds = load_dataset("hiyouga/math12k", split=split)
    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in ds]
    gts = [ex["answer"] for ex in ds]
    return prompts, gts


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_with_vllm(
    llm,
    prompts: list[str],
    ground_truths: list[str],
    max_examples: int,
    max_tokens: int,
) -> dict[str, float]:
    from vllm import SamplingParams

    if not prompts:
        return {
            "accuracy": 0.0,
            "count_correct_both1": 0.0,
            "count_format1_answer0": 0.0,
            "count_format0_answer0": 0.0,
            "n_examples": 0.0,
        }

    n = min(max_examples, len(prompts)) if max_examples > 0 else len(prompts)
    prompts = prompts[:n]
    ground_truths = ground_truths[:n]

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)

    correct = 0
    cat_110 = 0
    cat_100 = 0
    cat_000 = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        fr, ar = reward["format_reward"], reward["answer_reward"]
        correct += int(reward["reward"])
        if fr == 1.0 and ar == 1.0:
            cat_110 += 1
        elif fr == 1.0 and ar == 0.0:
            cat_100 += 1
        else:
            cat_000 += 1

    return {
        "accuracy": correct / len(outputs),
        "count_correct_both1": float(cat_110),
        "count_format1_answer0": float(cat_100),
        "count_format0_answer0": float(cat_000),
        "n_examples": float(len(outputs)),
    }


def iter_microbatches(rows: list[SFTRow], micro_batch_size: int):
    indices = list(range(len(rows)))
    random.shuffle(indices)
    for start in range(0, len(indices), micro_batch_size):
        batch_ids = indices[start : start + micro_batch_size]
        batch = [rows[i] for i in batch_ids]
        yield [b.prompt for b in batch], [b.response for b in batch]


def parse_size(size_str: str, total: int) -> int:
    if size_str.lower() == "full":
        return total
    return min(int(size_str), total)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-jsonl", default="data/intellect_math/train")
    parser.add_argument("--math-val-jsonl", default="data/intellect_math/dev")
    parser.add_argument("--math-test-jsonl", default="data/intellect_math/test")
    parser.add_argument("--intellect-test-path", default="data/intellect_math/test")
    parser.add_argument("--math-prompt-path", default="student/prompts/intellect.prompt")
    parser.add_argument("--dataset-size", default="full", help="int or 'full'")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=256)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/sft")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--save-every-eval", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sft_size{args.dataset_size}_lr{args.lr}_gbs{args.global_batch_size}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            config=vars(args),
        )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    train_rows = load_sft_train_rows(args.train_jsonl)
    keep_n = parse_size(args.dataset_size, len(train_rows))
    train_rows = train_rows[:keep_n]

    prompt_template = load_prompt_template(args.math_prompt_path)
    math_val_prompts, math_val_gts = load_math_split(args.math_val_jsonl, prompt_template, split="test")
    math_test_prompts, math_test_gts = load_math_split(args.math_test_jsonl, prompt_template, split="test")
    intellect_test_prompts, intellect_test_gts = load_intellect_split(args.intellect_test_path)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.policy_device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    grad_acc_steps = max(1, args.global_batch_size // args.micro_batch_size)
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "train_step",
                "eval_step",
                "train_loss",
                "eval_math_val_accuracy",
                "eval_intellect_test_accuracy",
                "eval_math_test_accuracy",
                "dataset_size",
                "lr",
                "global_batch_size",
                "micro_batch_size",
            ],
        )
        writer.writeheader()

    llm = init_vllm(
        model_id=args.model_id,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    global_micro_step = 0
    train_step = 0
    eval_step = 0
    best_val_acc = -1.0
    best_ckpt = None

    pbar = tqdm(total=args.max_train_steps if args.max_train_steps > 0 else None, desc="Training")
    for epoch in range(args.epochs):
        for prompt_batch, response_batch in iter_microbatches(train_rows, args.micro_batch_size):
            tokenized = tokenize_prompt_and_output(prompt_batch, response_batch, tokenizer)
            input_ids = tokenized["input_ids"].to(args.policy_device)
            labels = tokenized["labels"].to(args.policy_device)
            response_mask = tokenized["response_mask"].to(args.policy_device)

            out = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            policy_log_probs = out["log_probs"]
            normalize_constant = max(float(response_mask.sum().item()), 1.0)
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_acc_steps,
                normalize_constant=normalize_constant,
            )

            global_micro_step += 1
            if global_micro_step % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1
                pbar.update(1)

                train_loss = float(metadata["microbatch_loss"].item())
                if use_wandb:
                    wandb.log({"train_step": train_step, "train/loss": train_loss})

                with csv_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "train_step",
                        "eval_step",
                        "train_loss",
                        "eval_math_val_accuracy",
                        "eval_intellect_test_accuracy",
                        "eval_math_test_accuracy",
                        "dataset_size",
                        "lr",
                        "global_batch_size",
                        "micro_batch_size",
                    ])
                    writer.writerow({
                        "train_step": train_step,
                        "eval_step": "",
                        "train_loss": train_loss,
                        "eval_math_val_accuracy": "",
                        "eval_intellect_test_accuracy": "",
                        "eval_math_test_accuracy": "",
                        "dataset_size": keep_n,
                        "lr": args.lr,
                        "global_batch_size": args.global_batch_size,
                        "micro_batch_size": args.micro_batch_size,
                    })

                need_eval = (train_step % args.eval_every_steps == 0)
                if need_eval:
                    model.eval()
                    load_policy_into_vllm_instance(model, llm)

                    val_metrics = evaluate_with_vllm(
                        llm=llm,
                        prompts=math_val_prompts,
                        ground_truths=math_val_gts,
                        max_examples=args.max_eval_examples,
                        max_tokens=args.eval_max_tokens,
                    )
                    eval_step += 1

                    if use_wandb:
                        wandb.log(
                            {
                                "eval_step": eval_step,
                                "eval/math_val_accuracy": val_metrics["accuracy"],
                                "eval/math_val_count_correct_both1": val_metrics["count_correct_both1"],
                                "eval/math_val_count_format1_answer0": val_metrics["count_format1_answer0"],
                                "eval/math_val_count_format0_answer0": val_metrics["count_format0_answer0"],
                            }
                        )

                    with csv_path.open("a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            "train_step",
                            "eval_step",
                            "train_loss",
                            "eval_math_val_accuracy",
                            "eval_intellect_test_accuracy",
                            "eval_math_test_accuracy",
                            "dataset_size",
                            "lr",
                            "global_batch_size",
                            "micro_batch_size",
                        ])
                        writer.writerow({
                            "train_step": train_step,
                            "eval_step": eval_step,
                            "train_loss": "",
                            "eval_math_val_accuracy": val_metrics["accuracy"],
                            "eval_intellect_test_accuracy": "",
                            "eval_math_test_accuracy": "",
                            "dataset_size": keep_n,
                            "lr": args.lr,
                            "global_batch_size": args.global_batch_size,
                            "micro_batch_size": args.micro_batch_size,
                        })

                    if val_metrics["accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["accuracy"]
                        best_ckpt = ckpt_dir / "best"
                        model.save_pretrained(best_ckpt)
                        tokenizer.save_pretrained(best_ckpt)

                    if args.save_every_eval:
                        eval_ckpt = ckpt_dir / f"eval_step_{eval_step}"
                        model.save_pretrained(eval_ckpt)
                        tokenizer.save_pretrained(eval_ckpt)

                    model.train()

                if args.max_train_steps > 0 and train_step >= args.max_train_steps:
                    break

        if args.max_train_steps > 0 and train_step >= args.max_train_steps:
            break

    pbar.close()

    if best_ckpt is None:
        best_ckpt = ckpt_dir / "last"
        model.save_pretrained(best_ckpt)
        tokenizer.save_pretrained(best_ckpt)

    best_model = AutoModelForCausalLM.from_pretrained(
        str(best_ckpt),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.policy_device)
    best_model.eval()
    load_policy_into_vllm_instance(best_model, llm)

    intellect_test_metrics = evaluate_with_vllm(
        llm=llm,
        prompts=intellect_test_prompts,
        ground_truths=intellect_test_gts,
        max_examples=args.max_eval_examples,
        max_tokens=args.eval_max_tokens,
    )
    math_test_metrics = evaluate_with_vllm(
        llm=llm,
        prompts=math_test_prompts,
        ground_truths=math_test_gts,
        max_examples=args.max_eval_examples,
        max_tokens=args.eval_max_tokens,
    )

    summary = {
        "run_name": run_name,
        "dataset_size": keep_n,
        "lr": args.lr,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "epochs": args.epochs,
        "train_steps": train_step,
        "best_val_accuracy": best_val_acc,
        "best_checkpoint": str(best_ckpt),
        "intellect_test": intellect_test_metrics,
        "math_test": math_test_metrics,
        "metrics_csv": str(csv_path),
    }
    save_json(run_dir / "summary.json", summary)

    print("\n=== SFT RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if use_wandb:
        wandb.log(
            {
                "eval_step": eval_step + 1,
                "eval/intellect_test_accuracy": intellect_test_metrics["accuracy"],
                "eval/math_test_accuracy": math_test_metrics["accuracy"],
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
