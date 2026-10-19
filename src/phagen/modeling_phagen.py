# Adapted from https://github.com/lucidrains/MEGABYTE-pytorch
# Copyright (c) 2023 Phil Wang, MIT License

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from functools import wraps
from packaging import version
from itertools import zip_longest
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

import torch
from torch import nn, einsum
from torch.amp import autocast
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nn.attention import SDPBackend

from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel, is_accelerate_available, is_deepspeed_zero3_enabled

from .configuration_phagen import PhaGenConfig
from .generation_utils import MDMGenerationMixin
from .embedding_utils import pool_hidden_states
from .sequence_utils import extract_contigs

if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        extract_model_from_parallel,
        get_balanced_memory,
        get_max_memory,
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
    )

    accelerate_version = version.parse(importlib.metadata.version("accelerate"))
    if accelerate_version >= version.parse("0.31"):
        from accelerate.utils.modeling import get_state_dict_from_offload

AutoConfig.register("phagen", PhaGenConfig)

# helpers
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)


# rotary positional embedding
class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return next(self.buffers()).device

    @autocast('cuda', enabled = False)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(pos, t):
    # Keep q/k dtype consistent with v under Flash SDPA.
    # Without this cast, pos is often fp32 and can upcast q/k to fp32
    # while v remains bf16, causing dtype mismatch in scaled_dot_product_attention.
    if pos.dtype != t.dtype:
        pos = pos.to(dtype=t.dtype)
    return t * pos.cos() + rotate_half(t) * pos.sin()

# norm

class RMSNorm(Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# helper classes

class FeedForward(Module):
    """SwiGLU FeedForward (same as LLaMA/Qwen2/Dream)"""
    def __init__(self, *, dim, mult = 4, dropout = 0.):
        super().__init__()
        hidden_dim = int(dim * mult * 2 / 3)
        self.norm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias = False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias = False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))

print_once = once(print)
class Attend(Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.flash = flash
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # default cpu attention configs
        self.attn_cfg = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        if not torch.cuda.is_available() or not flash:
            return
        
        # Any GPU with SM >= 8.0 supports Flash Attention 2 (A100, L20, H100, RTX 4090 and similar).
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major >= 8:
            print_once(f'{device_properties.name} GPU detected (SM {device_properties.major}.{device_properties.minor}), using flash attention if input tensor is on cuda')
            # Flash Attention does not accept a custom attn_mask; PyTorch falls back to EFFICIENT_ATTENTION when one is passed.
            self.attn_cfg = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        else:
            print_once(f'{device_properties.name} GPU detected (SM {device_properties.major}.{device_properties.minor}), using math or mem efficient attention if input tensor is on cuda')
            self.attn_cfg = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def flash_attn(self, q, k, v, mask = None, is_causal = False):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)
        # Polarity flip. Our convention is True = ignore (padding), while SDPA uses True = attend.
        if exists(mask):
            mask = ~mask
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.nn.attention.sdpa_kernel(self.attn_cfg):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = is_causal
            )
        return out

    def forward(self, q, k, v, mask = None, is_causal = False):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, is_causal = is_causal)
        
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        # causal mask
        if is_causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        # key-side attention mask (bool True means mask/ignore)
        if exists(mask) and not is_causal:
            if mask.ndim != 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask.expand(-1, q.shape[1], q.shape[2], -1)
            sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        return out

class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, rotary_emb = None, mask = None, is_causal = False):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask = mask, is_causal = is_causal)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        rel_pos = True,
        flash_attn = False
    ):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim_head) if rel_pos else None
        self.layers = ModuleList([])

        for _ in range(layers):
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None, is_causal = False):
        n = x.shape[-2]
        rotary_emb = self.rotary_emb(n) if exists(self.rotary_emb) else None

        for attn, ff in self.layers:
            x = attn(x, rotary_emb = rotary_emb, mask = mask, is_causal = is_causal) + x
            x = ff(x) + x

        return self.norm(x)

# main class
class PhaGenModel(PreTrainedModel, MDMGenerationMixin):
    config_class = PhaGenConfig
    # Declare attention backend capabilities for Transformers >= 4.57 checks.
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # Configure the depth, hidden dimension and sequence length of each stage.

        self.config = config
        assert isinstance(config.depth, tuple) and isinstance(config.max_seq_len, tuple)
        assert len(config.depth) == len(config.max_seq_len)
        self.stages = len(config.depth)
        config.dim = cast_tuple(config.dim, self.stages)  # expand dim into a tuple matching the length of depth
        assert len(config.dim) == self.stages # dim and depth must have the same length

        *_, fine_dim = config.dim  # Hidden dimension of the finest stage.
        self.max_seq_len = config.max_seq_len
        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim in config.dim]) # per-stage start tokens, randomly initialised
        self.pos_embs = ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(config.dim, config.max_seq_len)]) if config.pos_emb else None # disabled here: PhaGen uses rotary positions only

        # Build token_embs for each stage, starting from the raw token embedding.
        patch_size = 1
        self.token_embs = ModuleList([]) # per-stage token embeddings
        self.token_embs.append(nn.Embedding(config.num_tokens, fine_dim))
        # Build progressively coarser embeddings by grouping finer-stage tokens.
        for dim_out, seq_len in zip(reversed(config.dim[:-1]), reversed(config.max_seq_len[1:])):
            patch_size *= seq_len
            self.token_embs.append(nn.Sequential(
                nn.Embedding(config.num_tokens, fine_dim), # map each token to a fine_dim vector
                Rearrange('... r d -> ... (r d)'), # reshape it into an (r * d) vector
                nn.LayerNorm(patch_size * fine_dim), # layer norm
                nn.Linear(patch_size * fine_dim, dim_out), # linear projection to dim_out
                nn.LayerNorm(dim_out) # layer norm
            ))

        # Build the per-stage transformers and the projection layer between consecutive stages.
        self.transformers = ModuleList([])
        self.to_next_transformer_projections = ModuleList([])
        attn_impl = getattr(config, "_attn_implementation_internal", None) or getattr(config, "_attn_implementation", None)
        if attn_impl is None:
            attn_impl = getattr(config, "attn_implementation", None)
        use_sdpa_backend = attn_impl in ("flash_attention_2", "sdpa")

        # zip_longest pads the shorter lists out to the length of the longest one.
        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(config.dim, config.dim[1:], config.depth, config.max_seq_len[1:]):
            self.transformers.append(Transformer(
                dim = h_dim,
                layers = stage_depth,
                dim_head = config.dim_head,
                heads = config.heads,
                attn_dropout = config.attn_dropout,
                ff_dropout = config.ff_dropout,
                ff_mult = config.ff_mult,
                rel_pos = config.rel_pos,
                flash_attn = use_sdpa_backend
            ))
            proj = nn.Identity()
            if exists(next_h_dim):
                proj = nn.Sequential(
                    Rearrange('b ... d -> b (...) d'),
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange('b m (n d) -> (b m) n d', n = next_seq_len)
                )
            self.to_next_transformer_projections.append(proj)

        # Build the final head layer.
        self.to_logits = nn.Linear(fine_dim, config.num_tokens)

    @torch.inference_mode()
    def embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pad_attention_mode: str = "attend_pad",
    ) -> torch.Tensor:
        """Return paper-style all_concat_mean embeddings, [batch, 964].

        Use eval mode and tokenized DNA without automatically added BOS/EOS.
        ``attention_mask`` describes content (1) and right padding (0), and
        defaults to input_ids != pad_token_id. Each row is evaluated on its
        own smallest complete patch canvas, matching the paper's per-sequence
        extraction regardless of other sequences in the batch. No tokens are
        added, truncated or moved to another device.

        By default, internal PAD positions participate in attention as in the
        paper, but are excluded from pooling. ``mask_pad`` also hides them in
        attention. Valid learned stage start tokens participate in pooling.
        Returns float32 vectors on the model output device.
        """
        if self.training:
            raise ValueError("Call model.eval() before extracting sequence embeddings.")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a 2D [batch, sequence] integer tensor.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs.")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("Embedding inputs must contain nonempty sequences.")
        if pad_attention_mode not in ("attend_pad", "mask_pad"):
            raise ValueError("pad_attention_mode must be 'attend_pad' or 'mask_pad'.")
        pad_id = self.config.pad_token_id
        if pad_id is None:
            raise ValueError("config.pad_token_id must be set for embedding extraction.")
        if attention_mask is None:
            content_mask = input_ids != pad_id
        else:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same [batch, sequence] shape as input_ids.")
            if not ((attention_mask == 0) | (attention_mask == 1)).all():
                raise ValueError("attention_mask must contain only 0 and 1.")
            content_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        lengths = content_mask.sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("Every sequence must contain at least one content token.")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        if not torch.equal(content_mask, positions < lengths[:, None]):
            raise ValueError("Embedding inputs support right padding only; attention_mask cannot have internal gaps.")
        if ((input_ids == pad_id) & content_mask).any():
            raise ValueError("Content positions cannot contain PAD tokens.")
        if (((input_ids < 0) | (input_ids >= self.config.num_tokens)) & content_mask).any():
            raise ValueError("Content token IDs must be within the model vocabulary.")
        if (lengths > reduce_mult(self.max_seq_len)).any():
            raise ValueError("An input sequence exceeds the model's token canvas.")

        multiple = reduce_mult(self.max_seq_len[1:])
        vectors = []
        for row, length in enumerate(lengths.tolist()):
            ids = input_ids[row : row + 1, :length].long()
            if pad_attention_mode == "attend_pad":
                padded_length = length + remainder_to_mult(length, multiple)
                model_mask = torch.ones((1, padded_length), dtype=torch.long, device=ids.device)
            else:
                model_mask = None
            hidden = self(input_ids=ids, attention_mask=model_mask, is_causal=False).hidden_states
            vectors.append(pool_hidden_states(hidden, length, self.max_seq_len))
        return torch.cat(vectors, dim=0)

    @staticmethod
    def extract_contigs(token_ids, tokenizer, min_length: int = 1) -> list[str]:
        """Return complete ACGT-only BOS/EOS spans from one generated canvas.

        See sequence_utils.extract_contigs. The paper's retained cohort uses
        min_length=2000; the default keeps any nonempty complete span.
        """
        return extract_contigs(token_ids, tokenizer, min_length=min_length)

    def get_input_embeddings(self):
        in_embs = []
        for embs in self.token_embs:
            if isinstance(embs, nn.Sequential):
                in_embs.append(embs[0])
            else:
                in_embs.append(embs)
        return in_embs
    
    def set_input_embeddings(self, value, idx):
        assert idx < len(self.token_embs), "the idx must be less than the length of token_embs"
        if isinstance(self.token_embs[idx], nn.Sequential):
            self.token_embs[idx][0] = value
        else:
            self.token_embs[idx] = value
    
    def get_output_embeddings(self):
        return self.to_logits
    
    def set_output_embeddings(self, value):
        self.to_logits = value

    def enable_input_require_grads(self):
        """
        Enable gradients for one embedding layer or a sequence of embedding layers.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        # Handle both single embedding and list of embeddings
        input_embeddings = self.get_input_embeddings()
        
        if isinstance(input_embeddings, (list, tuple)):
            # If it's a list, register hook for each embedding layer
            self._require_grads_hooks = []
            for emb in input_embeddings:
                hook = emb.register_forward_hook(make_inputs_require_grads)
                self._require_grads_hooks.append(hook)
        else:
            # If it's a single embedding layer
            self._require_grads_hook = input_embeddings.register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """
        Removes the _require_grads_hook.
        """
        if hasattr(self, "_require_grads_hooks"):
            # Handle list of hooks
            for hook in self._require_grads_hooks:
                hook.remove()
            del self._require_grads_hooks
        elif hasattr(self, "_require_grads_hook"):
            # Handle single hook
            self._require_grads_hook.remove()
            del self._require_grads_hook

    def get_deepspeed_state(self):
        return is_deepspeed_zero3_enabled()

    def resize_token_embeddings(
        self,
        new_num_tokens: int|None = None,
        pad_to_multiple_of: int|None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        # Resize the embedding layer when the original vocab_size grows or shrinks.
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Gather partitioned embedding weights before reading the vocabulary size.
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            try:
                import deepspeed
            except ImportError:
                pass

            if isinstance(model_embeds, list):
                with deepspeed.zero.GatheredParameters(model_embeds[0].weight, modifier_rank=None):
                    vocab_size = model_embeds[0].weight.shape[0]
            elif isinstance(model_embeds, nn.Embedding):
                with deepspeed.zero.GatheredParameters(model_embeds.weight, modifier_rank=None):
                    vocab_size = model_embeds.weight.shape[0]
            else:
                raise ValueError(f"model_embeds must be a list of nn.Embedding or a single nn.Embedding, but got {type(model_embeds)}")
        else:
            if isinstance(model_embeds, list):
                vocab_size = model_embeds[0].weight.shape[0]
            elif isinstance(model_embeds, nn.Embedding):
                vocab_size = model_embeds.weight.shape[0]
            else:
                raise ValueError(f"model_embeds must be a list of nn.Embedding or a single nn.Embedding, but got {type(model_embeds)}")

        # Update base model and current model config.
        self.config.get_text_config().vocab_size = vocab_size
        self.vocab_size = vocab_size
        self.config.num_tokens = vocab_size

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        # Fetch the existing token embedding layer.
        old_embeddingss = self.get_input_embeddings()
        for idx, old_embeddings in enumerate(old_embeddingss):
            # Build the new token embedding layer.
            new_embeddings = self._get_resized_embeddings(
                old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing
            )
            if hasattr(old_embeddings, "_hf_hook"):
                hook = old_embeddings._hf_hook
                add_hook_to_module(new_embeddings, hook)
            old_embeddings_requires_grad = old_embeddings.weight.requires_grad
            new_embeddings.requires_grad_(old_embeddings_requires_grad)
            self.set_input_embeddings(new_embeddings, idx)
            is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None

            # Update new_num_tokens with the actual size of new_embeddings
            if pad_to_multiple_of is not None:
                if is_deepspeed_zero3_enabled() and not is_quantized:
                    try:
                        import deepspeed
                    except ImportError:
                        pass

                    with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                        new_num_tokens = new_embeddings.weight.shape[0]
                else:
                    new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if (
            self.get_output_embeddings() is not None
            and not self.config.get_text_config(decoder=True).tie_word_embeddings
        ):
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()


    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        prev_stage_tokens_repr = None

        for stage_start_tokens, transformer, proj in zip(self.start_tokens, self.transformers, self.to_next_transformer_projections):
            tokens = repeat(stage_start_tokens, 'd -> b 1 d', b = batch_size)

            if exists(prev_stage_tokens_repr):
                tokens = tokens + prev_stage_tokens_repr[..., :tokens.shape[-2], :]

            tokens = transformer(tokens)
            prev_stage_tokens_repr = proj(tokens)

        logits = self.to_logits(tokens)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
        )

    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: torch.Tensor | None = None, 
        labels: torch.LongTensor | None = None, 
        mask_ratio: torch.FloatTensor | None = None, 
        is_causal: bool = False, 
        **kwargs
    ):
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            raise ValueError("config.pad_token_id must be set for PhaGenModel forward.")

        # batch size of the input
        batch = input_ids.shape[0]
        # the input must be 2-dimensional
        assert input_ids.ndim ==2, "input_ids must be a 2D tensor for modelling text data"

        # an empty input takes the forward_empty branch
        if input_ids.numel() == 0:
            return self.forward_empty(batch)

        compute_loss = labels is not None

        # sequence length of the input
        seq_len = input_ids.shape[-1]
        # the largest patch size
        multiple_of = reduce_mult(self.max_seq_len[1:])
        # padding needed to reach a whole number of patches
        padding = remainder_to_mult(seq_len, multiple_of)
        # apply the padding
        input_ids = F.pad(input_ids, (0, padding), value = pad_token_id)
        # reshape the input to (batch, -1, *self.max_seq_len[1:])
        input_ids = input_ids.reshape(batch, -1, *self.max_seq_len[1:])


        # Build the attention mask that hides padding tokens under non-causal attention.
        if mask_ratio is not None: 
            is_causal = False
            mask_ratio = F.pad(mask_ratio, (0, padding), value = 0.)
            if padding > 0:
                for i in range(mask_ratio.shape[0]):
                    mask_ratio[i, -padding:] = mask_ratio[i, :-padding].mean()
            mask_ratio = mask_ratio.reshape(-1)
            
        # With no attention_mask supplied and no causal decoding, derive one from input_ids.
        if attention_mask is None and not is_causal:
            # Mask from input_ids, with 1 at every non-padding position.
            attention_mask = (input_ids.reshape(batch, -1) != pad_token_id).bool()  # pad token from config
        
        # Pad a supplied attention_mask.
        if attention_mask is not None and attention_mask.numel() != input_ids.numel(): 
            attention_mask = F.pad(attention_mask, (0, padding), value = 0).bool()  # cast to bool first, then pad those positions with 0
        
        # Optimisation. When every position is valid, as under packing where attention_mask is all ones,
        # set it to None so Flash Attention stays available; SDPA falls back to Efficient Attention once given an explicit mask.
        if attention_mask is not None and attention_mask.all():
            attention_mask = None

        if compute_loss:
            labels = F.pad(labels, (0, padding), value = self.config.IGNORE_INDEX)
            labels = labels.reshape(-1)


        b, *prec_dims, device = *input_ids.shape, input_ids.device
        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'


        # byte-level token and position embeddings
        tokens_at_stages = []
        
        # fetch the positional embedding, which is None when pos_emb is disabled
        pos_embs = default(self.pos_embs, (None,) * len(prec_dims))
        
        # Collect each stage's token and position embeddings into tokens_at_stages.
        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), reversed(pos_embs), self.token_embs):
            tokens = token_emb(input_ids)
            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions
            # insert at the front, so the later loop consumes them last in, first out
            tokens_at_stages.insert(0, tokens)
            input_ids = rearrange(input_ids, '... m n -> ... (m n)') if ind > 0 else input_ids
        # Each stage retains its hierarchical axes and appends its hidden dimension.
        # input_ids is flattened back to (batch, padded_sequence_length).
        
        
        # Run the forward pass across the stages.
        # Holds the representation carried over from the previous stage.
        prev_stage_tokens_repr = None
        # List collecting the embeddings.
        hidden_states = [] 
        # Walk the stages in order.
        # Build the per-stage key-side mask from attention_mask, only in the non-causal case.
        stage_masks_valid = None
        if not is_causal and attention_mask is not None:
            # base mask: True for valid tokens, reshaped onto the hierarchical grid (b, L0, L1, ..., Ls)
            mask_grid = attention_mask.view(batch, *prec_dims)
            S = len(prec_dims)
            stage_masks_valid = []
            for s in range(S):
                # Aggregate over the finer dimensions; a position is valid if any finer position is.
                if s < S - 1:
                    dims_to_reduce = tuple(range(1 + s + 1, 1 + S))
                    stage_mask_valid = mask_grid.any(dim=dims_to_reduce)
                else:
                    stage_mask_valid = mask_grid  # the finest stage needs no aggregation
                stage_masks_valid.append(stage_mask_valid.bool())
        for stage_index, (stage_start_tokens, stage_tokens, transformer, proj) in enumerate(zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections)):
            # Pack the batch and coarser axes while retaining this stage's sequence axis.
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            # Key mask for this stage, shaped (b*m, n) to match the packing, with a leading False for the start token so it is never hidden.
            key_mask = None
            if not is_causal and stage_masks_valid is not None:
                stage_mask_valid = stage_masks_valid[stage_index]
                # match the packing by flattening every dimension except the last
                stage_mask_valid = stage_mask_valid.reshape(-1, stage_mask_valid.shape[-1])  # (Bflat, n)
                # flip True = valid into True = hide
                key_mask = ~stage_mask_valid
                # prepend one False for the start token so it stays visible
                key_mask = torch.cat([torch.zeros(key_mask.shape[0], 1, dtype=torch.bool, device=key_mask.device), key_mask], dim=1)
            
            # repeat stage_start_tokens ready for concatenation
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])
            # concatenate this stage's start tokens
            stage_tokens = torch.cat((stage_start_tokens, stage_tokens), dim = -2)
            # add the representation carried over from the previous stage
            stage_tokens = stage_tokens + F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.) if exists(prev_stage_tokens_repr) else stage_tokens
            
            # run the transformer forward pass
            # is_causal True selects causal attention, otherwise attention is bidirectional
            # when is_causal is False, stage_mask hides the padding tokens
            attended = transformer(stage_tokens, mask = key_mask, is_causal = is_causal) 
            
            # unpack this stage's embeddings
            attended = unpack_one(attended, ps, '* n d')
            # store this stage's embeddings in the list
            hidden_states.append(attended)
            # project this stage's embeddings into the next stage
            prev_stage_tokens_repr = proj(attended[..., 1:, :])


        logits = self.to_logits(attended)
        logits = logits[..., 1:, :]
        logits = rearrange(logits, 'b ... c -> b (...) c')
        gen_logits = logits[:, :seq_len]

        loss = None
        if compute_loss:
            logits_flat = logits.reshape(-1, self.config.num_tokens)
            if mask_ratio is not None:
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.config.IGNORE_INDEX)
                loss = loss_fct(logits_flat, labels)
                loss_mask = labels != self.config.IGNORE_INDEX
                loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=self.config.IGNORE_INDEX)
                loss = loss_fct(logits_flat, labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=gen_logits,
            hidden_states=hidden_states
        )   


ModelClass = PhaGenModel

__all__ = ["PhaGenModel"]
