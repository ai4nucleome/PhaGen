"""Padding-aware all_concat_mean pooling used by the paper evaluations."""

from __future__ import annotations

import math

import torch


def pool_hidden_states(hidden_states, sequence_length: int, max_seq_len) -> torch.Tensor:
    """Pool one unpadded sequence, including valid learned stage start tokens.

    The model pads to a multiple of the product of the finer stage lengths.
    A coarse position is valid when any of its DNA positions is valid. A
    learned start token is pooled only if its group contains a valid position.
    This is the paper's ``all_concat_mean``, not ``all_concat_content_mean``.
    Returns a float32 tensor of shape [1, sum(stage dimensions)].
    """
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")
    max_seq_len = tuple(max_seq_len)
    if not max_seq_len or any(not isinstance(size, int) or size < 1 for size in max_seq_len):
        raise ValueError("max_seq_len must contain positive integer stage lengths.")
    if len(hidden_states) != len(max_seq_len):
        raise ValueError("Expected one hidden-state tensor per stage.")
    if sequence_length > math.prod(max_seq_len):
        raise ValueError("sequence_length exceeds the model's token canvas.")

    multiple = math.prod(max_seq_len[1:])
    padded_length = ((sequence_length + multiple - 1) // multiple) * multiple
    device = hidden_states[0].device
    valid = torch.arange(padded_length, device=device)[None, :] < sequence_length
    grid = valid.reshape(1, padded_length // multiple, *max_seq_len[1:])
    pooled = []
    for stage, hidden in enumerate(hidden_states):
        if stage < len(max_seq_len) - 1:
            content = grid.any(dim=tuple(range(stage + 2, len(max_seq_len) + 1)))
        else:
            content = grid
        mask = torch.cat((content.any(dim=-1, keepdim=True), content), dim=-1)
        if tuple(hidden.shape[:-1]) != tuple(mask.shape):
            raise ValueError(
                f"Stage {stage + 1} hidden-state shape {tuple(hidden.shape)} does not "
                f"match sequence_length={sequence_length} and max_seq_len={max_seq_len}."
            )
        dims = tuple(range(1, hidden.ndim - 1))
        weight = mask.to(device=hidden.device, dtype=torch.float32)
        # Float32 accumulation also keeps long half-precision inputs stable.
        mean = (hidden.float() * weight.unsqueeze(-1)).sum(dim=dims)
        mean = mean / weight.sum(dim=dims).clamp_min(1).unsqueeze(-1)
        pooled.append(mean)
    return torch.cat(pooled, dim=-1)
