"""SFT (Supervised Fine-Tuning) utilities."""

import torch
from transformers import PreTrainedTokenizerBase


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
