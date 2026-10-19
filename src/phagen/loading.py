"""Load checkpoint data with the model implementation installed in this package."""

from __future__ import annotations

from os import PathLike

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .modeling_phagen import PhaGenModel


DEFAULT_MODEL_ID = "YusenWithR/PhaGen"


def load_model(
    model_name_or_path: str | PathLike = DEFAULT_MODEL_ID,
    *,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    local_files_only: bool = False,
) -> PhaGenModel:
    """Load PhaGen weights and config from a local directory or the Hub.

    Uses this package's model code and PyTorch SDPA. Returns the model in eval
    mode on ``device``. Float32 is the default used by the paper embeddings;
    other dtypes can change numerical results. ``revision`` selects a Hub
    version, and ``local_files_only`` prevents network access.
    """
    model = PhaGenModel.from_pretrained(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    return model.to(device).eval()


def load_tokenizer(
    model_name_or_path: str | PathLike = DEFAULT_MODEL_ID,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> PreTrainedTokenizerBase:
    """Load the checkpoint's tokenizer without executing remote Python code."""
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )

