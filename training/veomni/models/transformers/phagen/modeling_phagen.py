# Adapted from https://github.com/lucidrains/MEGABYTE-pytorch
# Copyright (c) 2023 Phil Wang, MIT License

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from functools import wraps
from beartype import beartype
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
        
        # SM >= 8.0 的 GPU 均支持 Flash Attention 2（A100, L20, H100, RTX 4090 等）
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major >= 8:
            print_once(f'{device_properties.name} GPU detected (SM {device_properties.major}.{device_properties.minor}), using flash attention if input tensor is on cuda')
            # Flash Attention 不支持自定义 attn_mask，当传入 mask 时 PyTorch 会自动退回到 EFFICIENT_ATTENTION
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
        # Convert True=ignore masks to the True=attend convention used by SDPA.
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

    @beartype
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        self.config = config
        assert isinstance(config.depth, tuple) and isinstance(config.max_seq_len, tuple)
        assert len(config.depth) == len(config.max_seq_len)
        self.stages = len(config.depth)
        config.dim = cast_tuple(config.dim, self.stages)  # 将 dim 转换为与 depth 长度相同的 tuple
        assert len(config.dim) == self.stages # 确保 dim 的长度与 depth 的长度相同

        *_, fine_dim = config.dim # Feature dimension of the finest stage.
        self.max_seq_len = config.max_seq_len
        self.start_tokens = nn.ParameterList([nn.Parameter(torch.randn(h_dim)) for h_dim in config.dim]) # 构建每个 stage 的 start_tokens，初始化为随机值
        self.pos_embs = ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(config.dim, config.max_seq_len)]) if config.pos_emb else None

        # 为每个 stage 创建 token_embs，首先是原始的 token embedding
        patch_size = 1
        self.token_embs = ModuleList([]) # 构建每个 stage 的 token_embs
        self.token_embs.append(nn.Embedding(config.num_tokens, fine_dim))
        for dim_out, seq_len in zip(reversed(config.dim[:-1]), reversed(config.max_seq_len[1:])):
            patch_size *= seq_len
            self.token_embs.append(nn.Sequential(
                nn.Embedding(config.num_tokens, fine_dim), # 先把 token 映射为 fine_dim 维度的向量
                Rearrange('... r d -> ... (r d)'), # 再把向量 reshape 为 (r * d) 维度的向量
                nn.LayerNorm(patch_size * fine_dim), # 再进行 LayerNorm
                nn.Linear(patch_size * fine_dim, dim_out), # 再进行线性变换，将维度变换为 dim_out
                nn.LayerNorm(dim_out) # 再进行 LayerNorm
            ))

        # 构建每个 stage 的 transformer 和 每个 stage 之间的 projection layer
        self.transformers = ModuleList([])
        self.to_next_transformer_projections = ModuleList([])
        attn_impl = getattr(config, "_attn_implementation_internal", None) or getattr(config, "_attn_implementation", None)
        if attn_impl is None:
            attn_impl = getattr(config, "attn_implementation", None)
        use_sdpa_backend = attn_impl in ("flash_attention_2", "sdpa")

        # zip_longest 会按照最长的list来填充短的序列，进行 zip
        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(config.dim, config.dim[1:], config.depth, config.max_seq_len[1:]):
            self.transformers.append(Transformer(
                dim = h_dim,            
                layers = stage_depth,   
                dim_head = config.dim_head,    
                heads = config.heads,          
                attn_dropout = config.attn_dropout,
                ff_dropout = config.ff_dropout,   
                ff_mult = config.ff_mult,         
                rel_pos = config.rel_pos,          # Rotary positional embeddings.
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

        # 构建最后的 head layer
        self.to_logits = nn.Linear(fine_dim, config.num_tokens)

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
        Enables the gradients for the input embeddings.
        
        Supports one embedding layer or a sequence of embedding layers.
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
        # resize embedding layer，当增加/减少原本的 vocab_size 时
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Since we are basically reusing the same old embeddings with new weight values, gathering is required
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
        # 获取之前旧的 token embedding 层
        old_embeddingss = self.get_input_embeddings()
        for idx, old_embeddings in enumerate(old_embeddingss):
            # 计算新的 token embedding 层
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
        mask_ratio: torch.LongTensor | None = None, 
        is_causal: bool = False, 
        **kwargs
    ):
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            raise ValueError("config.pad_token_id must be set for PhaGen forward.")

        # 输入数据的 batch size
        batch = input_ids.shape[0]
        # 检查输入的维度是否为2
        assert input_ids.ndim ==2, "input_ids must be a 2D tensor for modelling text data"

        # 如果输入是空的，则进入 forward_empty 分支
        if input_ids.numel() == 0:
            return self.forward_empty(batch)

        if labels is None:
            labels = input_ids
            
        # 获取输入的序列长度
        seq_len = input_ids.shape[-1]
        # 获取一个最大的 patch size 的长度
        multiple_of = reduce_mult(self.max_seq_len[1:])
        # 计算构建整数个 patch 需要进行多少 padding
        padding = remainder_to_mult(seq_len, multiple_of)
        # 进行 padding
        input_ids = F.pad(input_ids, (0, padding), value = pad_token_id)
        # 将输入的序列 reshape 为 (batch, -1, *self.max_seq_len[1:])
        input_ids = input_ids.reshape(batch, -1, *self.max_seq_len[1:])


        # 构建 attention mask，用于在非因果注意力中屏蔽 padding tokens
        if mask_ratio is not None: 
            is_causal = False
            mask_ratio = F.pad(mask_ratio, (0, padding), value = 0.)
            if padding > 0:
                for i in range(mask_ratio.shape[0]):
                    mask_ratio[i, -padding:] = mask_ratio[i, :-padding].mean()
            mask_ratio = mask_ratio.reshape(-1)
            
        # 如果没有提供 attention_mask，同时也不是自回归的方式，那么从 input_ids 自动构建 attention_mask
        if attention_mask is None and not is_causal:
            # 从 input_ids 构建 mask: 非 padding token 的位置为 1
            attention_mask = (input_ids.reshape(batch, -1) != pad_token_id).bool()  # pad token from config
        
        # 如果提供了 attention_mask，进行 padding
        if attention_mask is not None and attention_mask.numel() != input_ids.numel(): 
            attention_mask = F.pad(attention_mask, (0, padding), value = 0).bool()  # 先转换为 bool，然后 padding 位置为 0
        
        # 优化：如果所有位置都有效（如 packing 策略下 attention_mask 全 1），
        # 设为 None 以启用 Flash Attention（SDPA 收到显式 mask 时会退回到 Efficient Attention）
        if attention_mask is not None and attention_mask.all():
            attention_mask = None
            
        labels = F.pad(labels, (0, padding), value = self.config.IGNORE_INDEX)
        labels = labels.reshape(-1)    


        b, *prec_dims, device = *input_ids.shape, input_ids.device
        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'


        # 进行byte level的token embedding和position embedding
        tokens_at_stages = []
        
        # Use the configured positional embeddings, if enabled.
        pos_embs = default(self.pos_embs, (None,) * len(prec_dims))
        
        # 得到每个 stage 原始输入的 token embedding 和 position embedding，存储到 tokens_at_stages 中
        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), reversed(pos_embs), self.token_embs):
            tokens = token_emb(input_ids)
            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions
            # 插入，在之后的遍历中，类似队列，先进后出
            tokens_at_stages.insert(0, tokens)
            input_ids = rearrange(input_ids, '... m n -> ... (m n)') if ind > 0 else input_ids
        
        
        # 开始进行层间的前向传播
        # 初始化一个变量，用于存储前一个 stage 的 tokens 的 representation
        prev_stage_tokens_repr = None
        # 创建存储 embeddings 的list
        hidden_states = [] 
        # 遍历每个 stage，进行前向传播       
        # 基于 attention_mask 构造逐层的 key-side 掩码（仅在非因果情况下）
        stage_masks_valid = None
        if not is_causal and attention_mask is not None:
            # base mask: True for valid tokens; reshape到分层网格 (b, L0, L1, ..., Ls)
            mask_grid = attention_mask.view(batch, *prec_dims)
            S = len(prec_dims)
            stage_masks_valid = []
            for s in range(S):
                # 需要在更细的维度上聚合（任何一个细粒度为真则该位置为真）
                if s < S - 1:
                    dims_to_reduce = tuple(range(1 + s + 1, 1 + S))
                    stage_mask_valid = mask_grid.any(dim=dims_to_reduce)
                else:
                    stage_mask_valid = mask_grid  # 最细层不聚合
                stage_masks_valid.append(stage_mask_valid.bool())
        for stage_index, (stage_start_tokens, stage_tokens, transformer, proj) in enumerate(zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections)):
            # Pack the batch and preceding hierarchy dimensions for this stage.
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            # 计算该 stage 的 key 掩码：形状 (b*m, n) 与 pack 对齐；为 start token 在前面补 False（不屏蔽）
            key_mask = None
            if not is_causal and stage_masks_valid is not None:
                stage_mask_valid = stage_masks_valid[stage_index]
                # 对齐打包：把除最后一维外都展平
                stage_mask_valid = stage_mask_valid.reshape(-1, stage_mask_valid.shape[-1])  # (Bflat, n)
                # True=有效 -> 转成 True=屏蔽
                key_mask = ~stage_mask_valid
                # 为 start token 头部补一个 False（不屏蔽）
                key_mask = torch.cat([torch.zeros(key_mask.shape[0], 1, dtype=torch.bool, device=key_mask.device), key_mask], dim=1)
            
            # 重复 stage_start_tokens，用于拼接
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])
            # concat start token，即拼接每个 stage 的 start tokens
            stage_tokens = torch.cat((stage_start_tokens, stage_tokens), dim = -2)
            # 加上前一个 stage 的 tokens 的 representation
            stage_tokens = stage_tokens + F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.) if exists(prev_stage_tokens_repr) else stage_tokens
            
            # 进行 transformer 的前向传播
            # is_causal 为 True 进行因果注意力，反之进行双向注意力
            # 当 is_causal=False 时，传递 stage_mask 来屏蔽 padding tokens
            attended = transformer(stage_tokens, mask = key_mask, is_causal = is_causal) 
            
            # 将每个 stage 的 embeddings 解包
            attended = unpack_one(attended, ps, '* n d')
            # 将每个 stage 的 embeddings 存储到 list 中
            hidden_states.append(attended)
            # 将每个 stage 的 embeddings 投影到下一个 stage
            prev_stage_tokens_repr = proj(attended[..., 1:, :])



        logits = self.to_logits(attended)
        logits = logits[..., 1:, :]
        logits = rearrange(logits, 'b ... c -> b (...) c')
        gen_logits = logits[:, :seq_len]
        logits = logits.reshape(-1, self.config.num_tokens)
        
        if mask_ratio is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.config.IGNORE_INDEX)
            loss = loss_fct(logits, labels)
            loss_mask = labels != self.config.IGNORE_INDEX
            loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=self.config.IGNORE_INDEX)
            loss = loss_fct(logits, labels)
        

         
        return CausalLMOutputWithPast(
            loss=loss,
            logits=gen_logits,
            hidden_states=hidden_states
        )   



ModelClass = PhaGenModel

__all__ = ["PhaGenModel"]
