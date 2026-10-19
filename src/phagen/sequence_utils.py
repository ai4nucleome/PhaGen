"""Extract complete DNA records from a PhaGen token canvas."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral


def extract_contigs(token_ids, tokenizer, min_length=1) -> list[str]:
    """Return ACGT strings from complete BOS/EOS spans in one token sequence.

    ``token_ids`` must be a one-dimensional integer sequence or an object such
    as a one-dimensional torch tensor with a ``tolist()`` method. No torch
    dependency is required. A nested BOS restarts the span; any non-ACGT token
    inside a span discards it. Unclosed, empty and short spans are discarded.
    ``min_length`` counts DNA bases, excluding the boundary tokens.

    Invalid input shapes, token types or length limits raise ``ValueError``.
    The tokenizer must provide distinct BOS/EOS IDs and single-base ACGT IDs.
    """
    if isinstance(min_length, bool) or not isinstance(min_length, Integral) or min_length < 1:
        raise ValueError("min_length must be a positive integer counting DNA bases.")

    if hasattr(token_ids, "ndim") and token_ids.ndim != 1:
        raise ValueError("token_ids must be a single one-dimensional token sequence.")
    if hasattr(token_ids, "tolist"):
        try:
            token_ids = token_ids.tolist()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("token_ids must convert to a one-dimensional integer sequence.") from exc
    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(token_ids, Sequence):
        raise ValueError("token_ids must be a single one-dimensional integer sequence.")
    if any(isinstance(token_id, bool) or not isinstance(token_id, Integral) for token_id in token_ids):
        raise ValueError("token_ids must contain only integer token IDs, without nested sequences.")

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if any(isinstance(token_id, bool) or not isinstance(token_id, Integral) for token_id in (bos_id, eos_id)):
        raise ValueError("tokenizer must provide integer bos_token_id and eos_token_id values.")
    if bos_id == eos_id:
        raise ValueError("tokenizer BOS and EOS token IDs must be distinct.")
    try:
        base_ids = [tokenizer.convert_tokens_to_ids(base) for base in "ACGT"]
    except (AttributeError, LookupError, TypeError, ValueError) as exc:
        raise ValueError("tokenizer must map each of A, C, G and T to its own token ID.") from exc
    special_ids = {
        token_id
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id", "unk_token_id")
        if isinstance(token_id := getattr(tokenizer, name, None), Integral)
    }
    if (
        any(isinstance(token_id, bool) or not isinstance(token_id, Integral) for token_id in base_ids)
        or len(set(base_ids)) != 4
        or set(base_ids) & special_ids
    ):
        raise ValueError("tokenizer must provide four distinct ACGT token IDs separate from special tokens.")
    base_by_id = dict(zip(base_ids, "ACGT"))

    contigs = []
    current = None
    for token_id in token_ids:
        if token_id == bos_id:
            current = []
        elif token_id == eos_id:
            if current is not None and len(current) >= min_length:
                contigs.append("".join(current))
            current = None
        elif current is not None:
            if token_id in base_by_id:
                current.append(base_by_id[token_id])
            else:
                current = None
    return contigs
