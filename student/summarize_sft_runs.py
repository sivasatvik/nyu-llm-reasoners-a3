from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/sft")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    summaries = []
    for summary_path in runs_dir.glob("*/summary.json"):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if not summaries:
        print(f"No summary.json files found under {runs_dir}")
        return

    summaries.sort(key=lambda x: x.get("best_val_accuracy", -1.0), reverse=True)

    print("dataset_size\tlr\tglobal_bs\tbest_val_acc\tintellect_test_acc\tmath_test_acc\trun_name")
    for s in summaries:
        print(
            f"{s.get('dataset_size')}\t"
            f"{s.get('lr')}\t"
            f"{s.get('global_batch_size')}\t"
            f"{s.get('best_val_accuracy', 0.0):.4f}\t"
            f"{s.get('intellect_test', {}).get('accuracy', 0.0):.4f}\t"
            f"{s.get('math_test', {}).get('accuracy', 0.0):.4f}\t"
            f"{s.get('run_name')}"
        )

    best = summaries[0]
    print("\n=== BEST RUN ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
