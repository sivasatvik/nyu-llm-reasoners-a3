"""GRPO (Group Relative Policy Optimization) utilities."""

from __future__ import annotations

from typing import Callable, Literal

import torch

from student.sft import masked_mean


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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the per-token policy-gradient loss.

    Args:
        raw_rewards_or_advantages: shape (batch_size, 1) — scalar reward or
            advantage for each rollout response.
        policy_log_probs: shape (batch_size, sequence_length).

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): per-token PG loss
        (negate log-prob weighted by advantage; minimising this maximises
        expected reward).
    """
    # Broadcast (batch_size, 1) over sequence_length dim
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the per-token GRPO-Clip loss.

    Args:
        advantages: shape (batch_size, 1) — per-example advantage A.
        policy_log_probs: shape (batch_size, sequence_length).
        old_log_probs: shape (batch_size, sequence_length).
        cliprange: clip parameter ε.

    Returns:
        loss: shape (batch_size, sequence_length) — per-token clipped PG loss.
        metadata: dict with "is_clipped" bool tensor of the same shape,
            True where the clipped objective was the binding constraint.
    """
    # Importance-sampling ratio r_t = π_θ(a|s) / π_old(a|s)
    ratio = torch.exp(policy_log_probs - old_log_probs)  # (batch, seq)

    # Unclipped and clipped surrogate objectives (negated for minimisation)
    unclipped = -advantages * ratio                                         # (batch, seq)
    clipped   = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # Take element-wise max (pessimistic bound)
    loss = torch.max(unclipped, clipped)

    # A token is "clipped" when the clipped term was the binding constraint
    is_clipped = clipped < unclipped

    return loss, {"is_clipped": is_clipped}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convenience wrapper that dispatches to the correct PG loss routine.

    Args:
        policy_log_probs: shape (batch_size, sequence_length).
        loss_type: one of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: shape (batch_size, 1); required for "no_baseline".
        advantages: shape (batch_size, 1); required for "reinforce_with_baseline"
            and "grpo_clip".
        old_log_probs: shape (batch_size, sequence_length); required for "grpo_clip".
        cliprange: clip parameter ε; required for "grpo_clip".

    Returns:
        loss: shape (batch_size, sequence_length) per-token loss.
        metadata: auxiliary statistics from the underlying routine.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for loss_type='no_baseline'"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for loss_type='reinforce_with_baseline'"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for loss_type='grpo_clip'"
        assert old_log_probs is not None, "old_log_probs required for loss_type='grpo_clip'"
        assert cliprange is not None, "cliprange required for loss_type='grpo_clip'"
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a GRPO microbatch.

    Args:
        policy_log_probs: shape (batch_size, sequence_length).
        response_mask: shape (batch_size, sequence_length); 1 for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        loss_type: one of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: shape (batch_size, 1); required for "no_baseline".
        advantages: shape (batch_size, 1); required for the other two modes.
        old_log_probs: shape (batch_size, sequence_length); required for "grpo_clip".
        cliprange: clip parameter ε; required for "grpo_clip".

    Returns:
        loss: scalar tensor (microbatch loss adjusted for gradient accumulation).
        metadata: dict with stats from the underlying loss call.
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Average over response tokens using the mask, then scale for grad accumulation.
    microbatch_loss = masked_mean(per_token_loss, response_mask)
    loss = microbatch_loss / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), {"microbatch_loss": microbatch_loss.detach(), **metadata}
