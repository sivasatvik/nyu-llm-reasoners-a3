"""Minimal evaluation script for MATH and Intellect test sets."""

import json
from pathlib import Path
from collections import Counter

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn, extract_answer


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def reward_category(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct_both1"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "format1_answer0"
    return "format0_answer0"


def evaluate(llm, prompts, ground_truths, dataset_name: str, log_dir: Path, print_examples: int = 10):
    """Run evaluation, print reward-category stats, and save per-example logs."""
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    correct = 0
    counts = Counter()
    rows = []

    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]
        category = reward_category(reward["format_reward"], reward["answer_reward"])
        counts[category] += 1
        rows.append(
            {
                "idx": i,
                "dataset": dataset_name,
                "category": category,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"],
                "ground_truth": ground_truths[i],
                "parsed_answer": extract_answer(text),
                "response": text,
                "prompt": prompts[i],
            }
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    all_rows_path = log_dir / f"{dataset_name.lower()}_all_generations.jsonl"
    with all_rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved per-example logs: {all_rows_path}")
    print("Category counts:")
    print(f"  correct_both1:   {counts['correct_both1']}")
    print(f"  format1_answer0: {counts['format1_answer0']}")
    print(f"  format0_answer0: {counts['format0_answer0']}")

    for category in ["correct_both1", "format1_answer0", "format0_answer0"]:
        examples = [r for r in rows if r["category"] == category][:print_examples]
        examples_path = log_dir / f"{dataset_name.lower()}_{category}_examples.json"
        with examples_path.open("w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(examples)} examples for {category}: {examples_path}")

    return correct / len(outputs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data/intellect_math/test")
    parser.add_argument("--log-dir", default="outputs/eval_logs")
    parser.add_argument("--print-examples", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(args.intellect_path)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, gts = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(
        llm,
        prompts,
        gts,
        dataset_name="intellect",
        log_dir=Path(args.log_dir),
        print_examples=args.print_examples,
    )
    print(f"Intellect Accuracy: {acc:.4f}")

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(
        llm,
        prompts,
        gts,
        dataset_name="math",
        log_dir=Path(args.log_dir),
        print_examples=args.print_examples,
    )
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
