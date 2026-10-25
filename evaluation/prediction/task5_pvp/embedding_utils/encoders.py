from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


MODEL_SPECS = {
    "phagen": ("PhaGen", "all_concat_mean", 964, "YusenWithR/PhaGen", None),
    "megadna": ("MegaDNA", "all_concat_mean", 964, None, None),
    "evo": ("Evo", "mean", 4096, "evo-1-131k-base", "blocks.3"),
    "evo2_1b": ("Evo2-1B", "mean", 1920, "evo2_1b_base", "blocks.15"),
    "evo2_7b": ("Evo2-7B", "mean", 4096, "evo2_7b", "blocks.26"),
    "ntv2_50m": ("NTv2-50M", "last_hidden_mean", 512, "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", None),
    "ntv2_100m": ("NTv2-100M", "last_hidden_mean", 512, "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", None),
    "ntv2_250m": ("NTv2-250M", "last_hidden_mean", 768, "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", None),
    "ntv2_500m": ("NTv2-500M", "last_hidden_mean", 1024, "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", None),
    "ntv3_8m": ("NTv3-8M", "last_hidden_mean", 256, "InstaDeepAI/NTv3_8M_pre", None),
    "ntv3_100m": ("NTv3-100M", "last_hidden_mean", 768, "InstaDeepAI/NTv3_100M_pre", None),
    "ntv3_650m": ("NTv3-650M", "last_hidden_mean", 1536, "InstaDeepAI/NTv3_650M_pre", None),
}
NT_VOCAB = ["**", "A", "T", "C", "G", "#"]


def hierarchical_mean(hidden_states):
    return torch.cat([hidden.detach().mean(dim=tuple(range(hidden.ndim - 1)))
                      for hidden in hidden_states], dim=-1)


def sequence_mean(hidden):
    return hidden[0].mean(dim=0).float().cpu().numpy()


def pooling_mask(attention_mask, tokenizer, input_ids, exclude_special):
    if not exclude_special:
        return attention_mask
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_ids.discard(getattr(tokenizer, "pad_token_id", None))
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special |= input_ids.eq(int(token_id))
    return attention_mask.masked_fill(special, 0)


def masked_mean(hidden, attention_mask):
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).detach().float().cpu().numpy()


class IdentityUnembed(torch.nn.Module):
    def unembed(self, hidden_states):
        return hidden_states


def select_evo2_device(device):
    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("Native Evo2 inference requires a CUDA device")
    index = requested.index if requested.index is not None else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [item.strip() for item in visible.split(",") if item.strip()] if visible is not None else None
    if devices is not None and index >= len(devices):
        raise ValueError(f"CUDA device index {index} is outside CUDA_VISIBLE_DEVICES")
    if torch.cuda.is_initialized():
        if index != 0 or devices is None or len(devices) != 1:
            raise RuntimeError("Select the Evo2 GPU in a fresh process before CUDA initializes")
        return "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[index] if devices is not None else str(index)
    return "cuda:0"


class FrozenEncoder:
    def __init__(self, model_key, device="cuda:0", model=None, model_path=None,
                 model_repo=None, allow_download=False, pool_special_tokens=None):
        self.key = model_key
        self.pool_special_tokens = pool_special_tokens
        self.device = select_evo2_device(device) if model_key.startswith("evo2") else device
        self.spec = MODEL_SPECS[model_key]
        self.layer = self.spec[4]
        self.capture = {}
        self.hook = None
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if model_repo:
            sys.path.insert(0, str(Path(model_repo).resolve()))
        name = model or self.spec[3]
        if model_key == "phagen":
            from phagen import load_model, load_tokenizer
            self.model = load_model(name, device=device, torch_dtype=torch.float32,
                                    local_files_only=not allow_download)
            self.tokenizer = load_tokenizer(name, local_files_only=not allow_download)
        elif model_key == "megadna":
            if not model_path or not Path(model_path).is_file():
                raise FileNotFoundError("MegaDNA requires --model-path and its installed package or --model-repo")
            self.model = torch.load(model_path, map_location=device, weights_only=False).to(device).eval()
        elif model_key == "evo":
            from evo import Evo
            self.wrapper = Evo(name)
            self.model, self.tokenizer = self.wrapper.model, self.wrapper.tokenizer
            self.model = self.model.to(device).eval()
            self.model.unembed = IdentityUnembed().to(device)
            def capture(_module, _inputs, output):
                self.capture["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()
            self.hook = dict(self.model.named_modules())[self.layer].register_forward_hook(capture)
        elif model_key.startswith("evo2"):
            from evo2 import Evo2
            kwargs = {"local_path": str(model_path)} if model_path else {}
            self.model = Evo2(name, **kwargs)
        else:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True,
                                                           local_files_only=not allow_download)
            self.model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True,
                torch_dtype=torch.float32, local_files_only=not allow_download).to(device).eval()

    @torch.inference_mode()
    def encode_batch(self, sequences):
        if self.key.startswith("ntv"):
            batch = self.tokenizer(sequences, add_special_tokens=False, padding=True,
                                   pad_to_multiple_of=128, return_tensors="pt")
            ids = batch["input_ids"].to(self.device)
            if "attention_mask" in batch:
                attention = batch["attention_mask"].to(self.device)
            elif self.tokenizer.pad_token_id is not None:
                attention = ids.ne(self.tokenizer.pad_token_id).long()
            else:
                attention = torch.ones_like(ids)
            exclude = (self.pool_special_tokens == "exclude" if self.pool_special_tokens is not None
                       else self.key.startswith("ntv2"))
            mask = pooling_mask(attention, self.tokenizer, ids, exclude)
            hidden = self.model(input_ids=ids, attention_mask=attention, output_hidden_states=True).hidden_states[-1]
            return masked_mean(hidden, mask)
        rows = []
        for sequence in sequences:
            if self.key == "phagen":
                tokens = self.tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
                ids = tokens["input_ids"].to(self.device)
                rows.append(self.model.embed(ids, pad_attention_mode="attend_pad").squeeze(0).float().cpu().numpy())
            elif self.key == "megadna":
                ids = torch.tensor([NT_VOCAB.index(base) if base in NT_VOCAB else 1 for base in sequence],
                                   dtype=torch.long, device=self.device).unsqueeze(0)
                rows.append(hierarchical_mean(self.model(ids, return_value="embedding")).float().cpu().numpy())
            elif self.key == "evo":
                ids = torch.tensor(self.tokenizer.tokenize(sequence), dtype=torch.long, device=self.device).unsqueeze(0)
                self.capture.clear()
                self.model(ids)
                rows.append(sequence_mean(self.capture["hidden"]))
            else:
                ids = torch.tensor(self.model.tokenizer.tokenize(sequence), dtype=torch.long, device=self.device).unsqueeze(0)
                _, embeddings = self.model(ids, return_embeddings=True, layer_names=[self.layer])
                rows.append(sequence_mean(embeddings[self.layer]))
        return np.stack(rows)

    def close(self):
        if self.hook is not None:
            self.hook.remove()
