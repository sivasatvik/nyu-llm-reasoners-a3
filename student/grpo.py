"""GRPO (Group Relative Policy Optimization) utilities."""

from __future__ import annotations

from typing import Callable

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized within groups.

    Args:
        reward_fn: Scores a response against a ground truth, returning a dict
            with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] of length rollout_batch_size
            (= n_prompts_per_rollout_batch * group_size).
        repeated_ground_truths: list[str] of length rollout_batch_size.
            The ground truth for each prompt is repeated group_size times.
        group_size: Number of responses per prompt/group.
        advantage_eps: Small constant to avoid division by zero.
        normalize_by_std: If True, divide normalized advantages by per-group
            std; otherwise only subtract the group mean.

    Returns:
        advantages: torch.Tensor of shape (rollout_batch_size,) — group-
            normalized rewards.
        raw_rewards: torch.Tensor of shape (rollout_batch_size,) — raw
            (unnormalized) rewards.
        metadata: dict[str, float] — summary statistics of the batch rewards.
    """
    rollout_batch_size = len(rollout_responses)

    # Collect scalar rewards for every (response, ground_truth) pair.
    rewards_list: list[float] = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, ground_truth)
        rewards_list.append(float(result["reward"]))

    raw_rewards = torch.tensor(rewards_list, dtype=torch.float32)  # (rollout_batch_size,)

    # Reshape into groups: (n_groups, group_size)
    n_groups = rollout_batch_size // group_size
    rewards_grouped = raw_rewards.view(n_groups, group_size)  # (n_groups, group_size)

    # Per-group mean and std
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)   # (n_groups, 1)
    group_std = rewards_grouped.std(dim=1, keepdim=True, correction=1)  # (n_groups, 1)

    # Normalize
    advantages_grouped = rewards_grouped - group_mean
    if normalize_by_std:
        advantages_grouped = advantages_grouped / (group_std + advantage_eps)

    advantages = advantages_grouped.view(rollout_batch_size)  # flatten

    metadata: dict[str, float] = {
        "mean_reward": float(raw_rewards.mean().item()),
        "std_reward": float(raw_rewards.std().item()),
        "max_reward": float(raw_rewards.max().item()),
        "min_reward": float(raw_rewards.min().item()),
        "mean_advantage": float(advantages.mean().item()),
        "std_advantage": float(advantages.std().item()),
    }

    return advantages, raw_rewards, metadata
