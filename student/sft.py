"""SFT (Supervised Fine-Tuning) utilities."""

import torch
from transformers import PreTrainedTokenizerBase


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of tensor considering only elements where mask == 1.

    Args:
        tensor: data to average.
        mask: same shape as tensor; 1 = include, 0 = exclude.
        dim: dimension to average over. If None, average all masked elements.

    Returns:
        Masked mean; shape follows tensor.mean(dim) semantics.
    """
    mask = mask.to(dtype=tensor.dtype)
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()  # nan when no elements are masked in
    count = mask.sum(dim=dim)
    return masked.sum(dim=dim) / count  # nan where count == 0


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Prompts are tokenized with ``add_special_tokens=True`` (so a BOS token is
    prepended by default).  Outputs are tokenized with
    ``add_special_tokens=False`` to avoid inserting a second BOS token.  The
    two token sequences are concatenated, right-padded to the length of the
    longest sequence in the batch, and then the standard language-model shift
    is applied:

    * ``input_ids``   — all tokens *except* the last  (the model inputs)
    * ``labels``      — all tokens *except* the first (the prediction targets)
    * ``response_mask`` — 1 exactly on the response tokens inside ``labels``

    Args:
        prompt_strs: list[str] - the prompt strings.
        output_strs: list[str] - the output strings.
        tokenizer: PreTrainedTokenizer - the tokenizer to use.

    Returns:
        dict[str, torch.Tensor] with keys:
            "input_ids"      - shape (batch_size, max_len - 1)
            "labels"         - shape (batch_size, max_len - 1)
            "response_mask"  - shape (batch_size, max_len - 1), dtype long
    """
    # Tokenize prompts (with BOS) and outputs (no extra BOS)
    prompt_ids_list: list[list[int]] = tokenizer(
        prompt_strs, add_special_tokens=True
    ).input_ids
    output_ids_list: list[list[int]] = tokenizer(
        output_strs, add_special_tokens=False
    ).input_ids

    # Concatenate and record per-example lengths
    combined_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    prompt_lens = [len(p) for p in prompt_ids_list]
    output_lens = [len(o) for o in output_ids_list]

    # Padding token (fall back to EOS if pad is not defined)
    pad_id: int = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Right-pad all sequences to max combined length
    max_len = max(len(ids) for ids in combined_ids_list)
    padded = [
        ids + [pad_id] * (max_len - len(ids)) for ids in combined_ids_list
    ]
    padded_tensor = torch.tensor(padded, dtype=torch.long)  # (batch, max_len)

    # LM-style shift: input_ids drops the last token, labels drops the first
    input_ids = padded_tensor[:, :-1]  # (batch, max_len - 1)
    labels = padded_tensor[:, 1:]      # (batch, max_len - 1)

    # Build response_mask over the *labels* tensor.
    # After the shift, the first response token appears at index (P - 1) in
    # labels, and the last at index (P + O - 2) inclusive, where P is the
    # prompt length and O is the output length.
    batch_size = len(prompt_lens)
    seq_len = max_len - 1
    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for i, (P, O) in enumerate(zip(prompt_lens, output_lens)):
        start = P - 1       # inclusive
        end = P + O - 1     # exclusive
        response_mask[i, start:end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get per-token entropy over the vocabulary dimension.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length) with entropy values.
    """
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - log_z.unsqueeze(-1))
    expected_logits = torch.sum(probs * logits, dim=-1)
    return log_z - expected_logits


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probs and optional token entropy.

    Args:
        model: Causal language model used for scoring.
        input_ids: Tensor of shape (batch_size, sequence_length).
        labels: Tensor of shape (batch_size, sequence_length).
        return_token_entropy: If True, also return per-token entropy.

    Returns:
        dict with key "log_probs" and, optionally, "token_entropy".
    """
    logits = model(input_ids=input_ids).logits
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.to(logits.device).long().unsqueeze(-1),
    ).squeeze(-1)

    outputs: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        outputs["token_entropy"] = compute_entropy(logits)
    return outputs


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum tensor values over masked positions and normalize by a constant.

    Args:
        tensor: Tensor to sum and normalize.
        mask: Boolean/int mask with same shape as tensor. Positions where mask == 1
            are included in the sum.
        normalize_constant: Constant divisor for normalization.
        dim: Dimension to sum along. If None, sum over all dimensions.

    Returns:
        Normalized masked sum.
    """
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    summed = torch.sum(masked_tensor, dim=dim)
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute one SFT microbatch loss/backward step.

    Args:
        policy_log_probs: (batch_size, sequence_length) token log-probs.
        response_mask: (batch_size, sequence_length) response-token mask.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: Constant used in masked normalization.

    Returns:
        (loss, metadata) where loss is scaled for gradient accumulation.
    """
    nll = -policy_log_probs
    per_example_loss = masked_normalize(
        tensor=nll,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )
    microbatch_loss = torch.mean(per_example_loss)
    loss = microbatch_loss / gradient_accumulation_steps
    loss.backward()
    metadata: dict[str, torch.Tensor] = {"microbatch_loss": microbatch_loss.detach()}
    return loss.detach(), metadata
