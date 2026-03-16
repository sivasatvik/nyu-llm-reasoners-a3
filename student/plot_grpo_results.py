"""Plot GRPO validation accuracy vs training steps and print example rollouts.

Usage:
    uv run python student/plot_grpo_results.py --runs-dir outputs/grpo
    uv run python student/plot_grpo_results.py --run-dir outputs/grpo/my_run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None  # type: ignore
    plt = None         # type: ignore


def read_csv(path: Path) -> dict[str, list]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    data: dict[str, list] = {k: [] for k in rows[0]} if rows else {}
    for row in rows:
        for k, v in row.items():
            data[k].append(_float(v))
    return data


def plot_run(run_dir: Path, ax_acc, ax_reward, label: str) -> None:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"  [skip] no metrics.csv in {run_dir}")
        return

    data = read_csv(csv_path)

    # Validation accuracy vs eval_step (rows where eval_accuracy is set)
    eval_pairs = [
        (row["train_step"], row["eval_accuracy"])
        for row in (
            dict(zip(data.keys(), vals))
            for vals in zip(*data.values())
        )
        if row.get("eval_accuracy") is not None
    ]
    if eval_pairs:
        steps, accs = zip(*eval_pairs)
        ax_acc.plot(steps, accs, marker="o", markersize=3, label=label)

    # Training reward vs train_step
    train_pairs = [
        (row["train_step"], row["train_mean_reward"])
        for row in (
            dict(zip(data.keys(), vals))
            for vals in zip(*data.values())
        )
        if row.get("train_mean_reward") is not None
    ]
    if train_pairs:
        steps, rewards = zip(*train_pairs)
        ax_reward.plot(steps, rewards, alpha=0.6, label=label)


def print_rollout_log(run_dir: Path, n_per_checkpoint: int = 2) -> None:
    log_path = run_dir / "rollout_log.json"
    if not log_path.exists():
        return
    log = json.loads(log_path.read_text(encoding="utf-8"))
    print(f"\n{'='*70}")
    print(f"Example rollouts from: {run_dir.name}")
    print(f"{'='*70}")
    for entry in log:
        step = entry["train_step"]
        print(f"\n--- train_step = {step} ---")
        for ex in entry["examples"][:n_per_checkpoint]:
            print(f"  Prompt (tail): {ex['prompt'][-120:]!r}")
            print(f"  GT:            {ex['gt']!r}")
            print(f"  Rollout:       {ex['rollout'][:300]!r}")
            print(f"  Reward:        {ex['reward']:.2f}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/grpo",
                        help="Directory containing multiple run subdirectories.")
    parser.add_argument("--run-dir",  default="",
                        help="Single run directory (overrides --runs-dir).")
    parser.add_argument("--output",   default="outputs/grpo_results.png")
    parser.add_argument("--n-rollout-examples", type=int, default=2)
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        base = Path(args.runs_dir)
        run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and (d / "metrics.csv").exists()])

    if not run_dirs:
        print("No runs found.")
        return

    # Print rollout logs regardless of matplotlib availability
    for run_dir in run_dirs:
        print_rollout_log(run_dir, n_per_checkpoint=args.n_rollout_examples)

    if plt is None:
        print("matplotlib not installed — skipping plot generation.")
        return

    fig, (ax_acc, ax_reward) = plt.subplots(1, 2, figsize=(14, 5))
    ax_acc.set_title("Validation Accuracy vs Training Steps")
    ax_acc.set_xlabel("Training step")
    ax_acc.set_ylabel("Accuracy")

    ax_reward.set_title("Mean Training Reward vs Training Steps")
    ax_reward.set_xlabel("Training step")
    ax_reward.set_ylabel("Mean reward (per rollout batch)")

    for run_dir in run_dirs:
        plot_run(run_dir, ax_acc, ax_reward, label=run_dir.name)

    for ax in (ax_acc, ax_reward):
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
