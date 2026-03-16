"""GRPO training loop for Countdown (and optionally intellect_math).

Usage (single GPU, default settings):
    uv run python student/train_grpo.py

Key algorithm (per training step):
    1.  Sample n_prompts_per_step prompts from the training set.
    2.  Generate group_size rollouts per prompt via vLLM (temperature sampling).
    3.  Score every rollout → raw rewards.
    4.  Compute group-normalized advantages via compute_group_normalized_rewards.
    5.  Optionally compute old_log_probs from the frozen policy (grpo_clip).
    6.  Iterate over microbatches, call grpo_microbatch_train_step, accumulate grads.
    7.  Clip gradients, optimizer step, zero grads.
    8.  Periodically evaluate (greedy decoding) and log.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import torch
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from student.drgrpo_grader import question_only_reward_fn
from student.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from student.sft import get_response_log_probs, tokenize_prompt_and_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class CountdownExample:
    prompt: str
    ground_truth: str   # the target number as a string
    numbers: list[int]
    target: int


def load_countdown_split(
    path: str | Path,
    prompt_template: str,
) -> list[CountdownExample]:
    """Load a Countdown dataset split (HuggingFace disk format or JSONL)."""
    p = Path(path)
    if p.is_dir():
        ds = load_from_disk(str(p))
        rows = list(ds)
    else:
        rows = read_jsonl(p)

    examples: list[CountdownExample] = []
    for row in rows:
        # Support both field-name conventions seen in the wild
        nums   = row.get("nums", row.get("numbers", []))
        target = row.get("target", row.get("answer", 0))
        question = f"Using the numbers {nums}, reach the target {target}."
        prompt = prompt_template.replace("{question}", question)
        examples.append(CountdownExample(
            prompt=prompt,
            ground_truth=str(target),
            numbers=nums,
            target=target,
        ))
    return examples


# ---------------------------------------------------------------------------
# vLLM helpers (mirrors train_sft.py)
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch  = patch(
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
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


@contextlib.contextmanager
def vllm_generate_context(
    model_id: str,
    policy: PreTrainedModel,
    policy_device: str,
    vllm_device: str,
    seed: int,
    gpu_memory_utilization: float,
):
    """Offload policy to CPU on single-GPU, create vLLM, yield, destroy."""
    single_gpu = policy_device == vllm_device
    if single_gpu:
        policy.to("cpu")
        torch.cuda.empty_cache()

    llm = init_vllm(model_id, vllm_device, seed, gpu_memory_utilization)
    load_policy_into_vllm_instance(policy, llm)
    try:
        yield llm
    finally:
        del llm
        torch.cuda.empty_cache()
        if single_gpu:
            policy.to(policy_device)


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def generate_rollouts(
    llm,
    prompts: list[str],
    group_size: int,
    temperature: float,
    max_tokens: int,
) -> list[list[str]]:
    """Return a list-of-lists: [group_size rollouts per prompt]."""
    from vllm import SamplingParams

    # Repeat each prompt group_size times
    repeated = [p for p in prompts for _ in range(group_size)]
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1)
    outputs = llm.generate(repeated, params)
    texts = [o.outputs[0].text for o in outputs]

    # Group back
    groups: list[list[str]] = []
    for i in range(0, len(texts), group_size):
        groups.append(texts[i:i + group_size])
    return groups


def greedy_generate(llm, prompts: list[str], max_tokens: int) -> list[str]:
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def microbatch_iter(
    prompts: list[str],
    rollouts: list[str],
    repeated_gts: list[str],
    advantages: torch.Tensor,
    raw_rewards: torch.Tensor,
    old_log_probs_flat: torch.Tensor | None,
    micro_batch_size: int,
):
    """Yield microbatches of (prompts, rollouts, gts, advantages slice, ...)."""
    n = len(prompts)
    indices = list(range(n))
    for start in range(0, n, micro_batch_size):
        idx = indices[start:start + micro_batch_size]
        yield (
            [prompts[i] for i in idx],
            [rollouts[i] for i in idx],
            [repeated_gts[i] for i in idx],
            advantages[idx].unsqueeze(-1),   # (mb, 1)
            raw_rewards[idx],
            old_log_probs_flat[idx] if old_log_probs_flat is not None else None,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training on Countdown")
    # Data
    parser.add_argument("--train-path",   default="data/countdown/dataset/train")
    parser.add_argument("--val-path",     default="data/countdown/dataset/dev")
    parser.add_argument("--prompt-path",  default="student/prompts/countdown.prompt")
    # Model
    parser.add_argument("--model-id",     default="Qwen/Qwen2.5-Math-1.5B")
    # GRPO hyper-parameters
    parser.add_argument("--loss-type",    default="grpo_clip",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--group-size",   type=int,   default=8)
    parser.add_argument("--cliprange",    type=float, default=0.2)
    parser.add_argument("--normalize-by-std", action="store_true", default=True)
    parser.add_argument("--advantage-eps",    type=float, default=1e-6)
    # Batching
    parser.add_argument("--n-prompts-per-step",  type=int, default=8,
                        help="Number of distinct prompts to sample per training step.")
    parser.add_argument("--micro-batch-size",    type=int, default=4)
    parser.add_argument("--global-batch-size",   type=int, default=0,
                        help="If 0, defaults to n_prompts_per_step * group_size.")
    parser.add_argument("--train-steps",         type=int, default=500)
    # Optimiser
    parser.add_argument("--lr",           type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--max-seq-len",  type=int,   default=1024)
    # Generation
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--max-gen-tokens",      type=int,   default=512)
    # Eval
    parser.add_argument("--eval-every",       type=int, default=50)
    parser.add_argument("--eval-prompts",     type=int, default=64)
    parser.add_argument("--n-example-rollouts", type=int, default=3,
                        help="Number of example rollouts to print/log each eval.")
    # Devices
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device",   default="",
                        help="Defaults to cuda:1 if 2+ GPUs, else cuda:0.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    # Logging
    parser.add_argument("--output-dir",    default="outputs/grpo")
    parser.add_argument("--run-name",      default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity",  default="")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    # ---- device resolution ------------------------------------------------
    if not args.vllm_device:
        args.vllm_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    rollout_batch_size = args.n_prompts_per_step * args.group_size
    global_batch_size  = args.global_batch_size or rollout_batch_size
    grad_acc_steps     = max(1, global_batch_size // args.micro_batch_size)

    # ---- run directory ----------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"grpo_{args.loss_type}_gs{args.group_size}_{timestamp}"
    run_dir   = Path(args.output_dir) / run_name
    ckpt_dir  = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- W&B ---------------------------------------------------------------
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
        wandb.define_metric("eval/*",  step_metric="eval_step")

    # ---- CSV ---------------------------------------------------------------
    csv_path = run_dir / "metrics.csv"
    csv_fieldnames = [
        "train_step", "eval_step",
        "train_loss", "train_mean_reward", "train_mean_advantage",
        "eval_accuracy", "eval_correct", "eval_format_only", "eval_neither",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_fieldnames).writeheader()

    def write_csv_row(row: dict) -> None:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writerow({k: row.get(k, "") for k in csv_fieldnames})

    # ---- data --------------------------------------------------------------
    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    train_examples  = load_countdown_split(args.train_path, prompt_template)
    val_examples    = load_countdown_split(args.val_path,   prompt_template)
    print(f"Train: {len(train_examples)} examples  |  Val: {len(val_examples)} examples")

    # ---- model & tokenizer -------------------------------------------------
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(args.policy_device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- training loop -----------------------------------------------------
    train_step = 0
    eval_step  = 0
    best_val_acc = -1.0
    best_ckpt    = None

    # Rollout log kept for the report
    rollout_log: list[dict] = []

    pbar = tqdm(total=args.train_steps, desc="GRPO steps")

    while train_step < args.train_steps:

        # 1. Sample prompts
        sampled = random.choices(train_examples, k=args.n_prompts_per_step)
        prompts_batch       = [ex.prompt        for ex in sampled]
        ground_truths_batch = [ex.ground_truth  for ex in sampled]

        # 2. Generate rollouts via vLLM
        model.eval()
        with vllm_generate_context(
            model_id=args.model_id,
            policy=model,
            policy_device=args.policy_device,
            vllm_device=args.vllm_device,
            seed=args.seed + train_step,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        ) as llm:
            rollout_groups = generate_rollouts(
                llm=llm,
                prompts=prompts_batch,
                group_size=args.group_size,
                temperature=args.rollout_temperature,
                max_tokens=args.max_gen_tokens,
            )
        model.train()

        # Flatten lists: (n_prompts * group_size,)
        flat_prompts  = [p for p, g in zip(prompts_batch, rollout_groups) for _ in g]
        flat_rollouts = [r for g in rollout_groups     for r in g]
        flat_gts      = [gt for gt, g in zip(ground_truths_batch, rollout_groups) for _ in g]

        # 3. Rewards + group-normalized advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=question_only_reward_fn,
            rollout_responses=flat_rollouts,
            repeated_ground_truths=flat_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.normalize_by_std,
        )
        advantages  = advantages.to(args.policy_device)
        raw_rewards = raw_rewards.to(args.policy_device)

        # 4. Optionally compute old_log_probs (needed for grpo_clip)
        old_log_probs_flat: torch.Tensor | None = None
        if args.loss_type == "grpo_clip":
            lp_list: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(flat_prompts), args.micro_batch_size):
                    p_mb  = flat_prompts[start:start + args.micro_batch_size]
                    r_mb  = flat_rollouts[start:start + args.micro_batch_size]
                    tok   = tokenize_prompt_and_output(p_mb, r_mb, tokenizer)
                    T     = args.max_seq_len
                    ids   = tok["input_ids"][:, :T].to(args.policy_device)
                    lbs   = tok["labels"][:, :T].to(args.policy_device)
                    out   = get_response_log_probs(
                        model=model, input_ids=ids, labels=lbs, return_token_entropy=False
                    )
                    lp = out["log_probs"].detach().cpu()  # (mb, seq_mb)
                    # Pad to T so all microbatches share the same seq dimension.
                    if lp.shape[1] < T:
                        lp = torch.nn.functional.pad(lp, (0, T - lp.shape[1]), value=0.0)
                    lp_list.append(lp)
            old_log_probs_flat = torch.cat(lp_list, dim=0)  # (rollout_bs, T)

        # 5. Gradient-accumulation update
        optimizer.zero_grad(set_to_none=True)
        accum_loss   = 0.0
        micro_count  = 0

        for (
            mb_prompts, mb_rollouts, mb_gts,
            mb_adv, mb_raw,
            mb_old_lp,
        ) in microbatch_iter(
            flat_prompts, flat_rollouts, flat_gts,
            advantages, raw_rewards,
            old_log_probs_flat,
            args.micro_batch_size,
        ):
            tok          = tokenize_prompt_and_output(mb_prompts, mb_rollouts, tokenizer)
            T            = args.max_seq_len
            input_ids    = tok["input_ids"][:, :T].to(args.policy_device)
            labels       = tok["labels"][:, :T].to(args.policy_device)
            response_mask = tok["response_mask"][:, :T].to(args.policy_device)

            out = get_response_log_probs(
                model=model, input_ids=input_ids, labels=labels, return_token_entropy=False
            )
            policy_log_probs = out["log_probs"]

            # Align old_log_probs to the same (possibly truncated) seq length
            if mb_old_lp is not None:
                old_lp = mb_old_lp[:, :policy_log_probs.shape[1]].to(args.policy_device)
            else:
                old_lp = None

            loss, _meta = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_acc_steps,
                loss_type=args.loss_type,
                raw_rewards=mb_raw.unsqueeze(-1) if args.loss_type == "no_baseline" else None,
                advantages=mb_adv if args.loss_type != "no_baseline" else None,
                old_log_probs=old_lp,
                cliprange=args.cliprange,
            )
            accum_loss  += float(loss.item())
            micro_count += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        train_step += 1
        pbar.update(1)

        mean_loss    = accum_loss / max(micro_count, 1)
        mean_reward  = float(reward_meta["mean_reward"])
        mean_adv     = float(reward_meta["mean_advantage"])

        if use_wandb:
            import wandb
            wandb.log({
                "train_step":         train_step,
                "train/loss":         mean_loss,
                "train/mean_reward":  mean_reward,
                "train/mean_advantage": mean_adv,
                "train/max_reward":   float(reward_meta["max_reward"]),
                "train/min_reward":   float(reward_meta["min_reward"]),
            })

        write_csv_row({
            "train_step":        train_step,
            "train_loss":        mean_loss,
            "train_mean_reward": mean_reward,
            "train_mean_advantage": mean_adv,
        })

        # 6. Periodic eval & rollout logging
        if train_step % args.eval_every == 0:
            eval_prompts = [ex.prompt       for ex in val_examples[:args.eval_prompts]]
            eval_gts     = [ex.ground_truth for ex in val_examples[:args.eval_prompts]]

            model.eval()
            with vllm_generate_context(
                model_id=args.model_id,
                policy=model,
                policy_device=args.policy_device,
                vllm_device=args.vllm_device,
                seed=args.seed,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            ) as llm:
                eval_texts = greedy_generate(llm, eval_prompts, max_tokens=args.max_gen_tokens)
            model.train()

            correct = 0; fmt_only = 0; neither = 0
            for txt, gt in zip(eval_texts, eval_gts):
                r = question_only_reward_fn(txt, gt)
                fr, ar = r["format_reward"], r["answer_reward"]
                if fr == 1.0 and ar == 1.0:
                    correct += 1
                elif fr == 1.0:
                    fmt_only += 1
                else:
                    neither += 1

            acc = correct / len(eval_texts)
            eval_step += 1
            print(
                f"\n[eval step {eval_step} | train step {train_step}] "
                f"acc={acc:.3f}  correct={correct}  fmt_only={fmt_only}  neither={neither}"
            )

            # Log a few example rollouts from the current training batch
            print(f"\n--- Example rollouts (train_step={train_step}) ---")
            for i in range(min(args.n_example_rollouts, len(flat_rollouts))):
                reward_val = float(raw_rewards[i].item())
                print(f"  [Prompt #{i}] {flat_prompts[i][-120:]!r}")
                print(f"  [GT]      {flat_gts[i]!r}")
                print(f"  [Rollout] {flat_rollouts[i][:300]!r}")
                print(f"  [Reward]  {reward_val:.2f}\n")
            rollout_log.append({
                "train_step": train_step,
                "examples": [
                    {
                        "prompt":  flat_prompts[i][-300:],
                        "gt":      flat_gts[i],
                        "rollout": flat_rollouts[i][:600],
                        "reward":  float(raw_rewards[i].item()),
                    }
                    for i in range(min(args.n_example_rollouts, len(flat_rollouts)))
                ],
            })

            if use_wandb:
                import wandb
                wandb.log({
                    "eval_step":        eval_step,
                    "eval/accuracy":    acc,
                    "eval/correct":     correct,
                    "eval/fmt_only":    fmt_only,
                    "eval/neither":     neither,
                })

            write_csv_row({
                "eval_step":    eval_step,
                "eval_accuracy": acc,
                "eval_correct":  correct,
                "eval_format_only": fmt_only,
                "eval_neither":  neither,
            })

            if acc > best_val_acc:
                best_val_acc = acc
                best_ckpt    = ckpt_dir / "best"
                model.save_pretrained(best_ckpt)
                tokenizer.save_pretrained(best_ckpt)
                print(f"  ✓ new best val acc = {best_val_acc:.3f} → saved to {best_ckpt}")

    pbar.close()

    # ---- save final checkpoint & rollout log -------------------------------
    final_ckpt = ckpt_dir / "final"
    model.save_pretrained(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)

    save_json(run_dir / "rollout_log.json", rollout_log)
    summary = {
        "run_name":      run_name,
        "train_steps":   train_step,
        "best_val_acc":  best_val_acc,
        "best_ckpt":     str(best_ckpt),
        "final_ckpt":    str(final_ckpt),
        "metrics_csv":   str(csv_path),
        "rollout_log":   str(run_dir / "rollout_log.json"),
        "args":          vars(args),
    }
    save_json(run_dir / "summary.json", summary)
    print("\n=== GRPO SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2))

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
