# Adapted from https://github.com/pengzhangzhi/Open-dLLM
# Licensed under the Apache License, Version 2.0

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)

# origin reveals randomly selected masked positions.
# entropy, topk_margin and maskgit_plus rank masked positions by confidence.
# p2 can re-mask generated positions; supplied prompt tokens remain fixed.


def top_p_logits(logits, top_p=None):
    if top_p is None or top_p >= 1.0:
        return logits
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"`top_p` must be in (0, 1], got {top_p}")

    # top-p sampling
    ## Keep the smallest set of tokens whose cumulative probability reaches top_p and send the rest to negative infinity.
    
    # 1. Sort the logits.
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    # 2. Take softmax probabilities, then their cumulative sum.
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)
    # 3. Find where the cumulative probability exceeds top_p.
    sorted_indices_to_remove = cumulative_probs > top_p
    # 4. Shift the mask so at least one token always survives.
    #        Shifting the cumulative probability right by one guarantees a sampleable token for any top_p.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    # 5. Build the mask and apply it to the original logits.
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(int(top_k), logits.size(-1))
    # torch.topk(logits, top_k)[0] gives the values of the top_k elements.
    # torch.topk(logits, top_k)[0][..., -1, None] gives the smallest of those top_k values.
    # logits < that value then marks every position below the top_k threshold.
    indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    # Apply the mask.
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def restrict_logits_to_allowed_tokens(logits, allowed_token_ids=None):
    if allowed_token_ids is None:
        return logits
    allowed_token_ids = sorted({int(token_id) for token_id in allowed_token_ids})
    if not allowed_token_ids:
        raise ValueError("`allowed_token_ids` must not be empty when provided.")
    if allowed_token_ids[0] < 0 or allowed_token_ids[-1] >= logits.size(-1):
        raise ValueError(
            "`allowed_token_ids` contains an ID outside the model vocabulary: "
            f"ids={allowed_token_ids}, vocab_size={logits.size(-1)}"
        )
    blocked = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
    blocked[torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)] = False
    return logits.masked_fill(blocked, float("-inf"))


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, alg="origin"):
    if logits.size(-1) == 0:
        raise ValueError("`logits` must have a non-empty vocabulary dimension.")

    if temperature > 0:
        logits = logits / temperature
    logits = top_p_logits(logits, top_p)
    if top_k is not None and top_k > 0:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        # Sample from the distribution.
        x0 = dists.Categorical(probs=probs).sample()
    else:
        # Greedy selection.
        _, x0 = probs.max(dim=-1)
    # Gather the softmax probability of the sampled token index and use it as the model's confidence at that position.
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        if probs.size(-1) < 2:
            raise ValueError("`topk_margin` requires vocabulary size >= 2.")
        # Probability gap between the most likely and the second most likely token.
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    elif alg == "entropy":
        # Negative Shannon entropy: higher values indicate greater confidence.
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)
    elif alg in ["maskgit_plus", "origin", "p2"]:
        pass
    else:
        raise NotImplementedError(f"Algorithm {alg} not implemented.")
    
    return confidence, x0


@dataclass
class MDMModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.LongTensor, ...]] = None

class MDMGenerationConfig(GenerationConfig):
    """
    Parameters (MDM-specific, popped before forwarding to GenerationConfig):
        mask_token_id, steps, alg, alg_temp, block_pad_token,
        allowed_token_ids,
        output_history, pad_attention_mode, eps

    Parameters (inherited from GenerationConfig, passed through via super()):
        pad_token_id, max_new_tokens, temperature, top_p, top_k,
        num_return_sequences, return_dict_in_generate, ...
    """

    def __init__(self, **kwargs):
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = True

        # --- MDM-specific (not recognized by GenerationConfig) ---
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "entropy")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", 0.0)
        self.block_pad_token: bool = kwargs.pop("block_pad_token", False)
        self.allowed_token_ids: Optional[list[int]] = kwargs.pop("allowed_token_ids", None)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.pad_attention_mode: str = kwargs.pop("pad_attention_mode", "attend_pad")
        self.eps: float = kwargs.pop("eps", 1e-3)

        # --- Inherited (handled by GenerationConfig.__init__) ---
        # pad_token_id:            int | None   (default: None)
        # max_new_tokens:          int | None   (default: None)
        # temperature:             float        (default: 1.0)
        # top_p:                   float | None (default: None)
        # top_k:                   int | None   (default: None)
        # num_return_sequences:    int          (default: 1)
        # return_dict_in_generate: bool         (default: False)

        super().__init__(**kwargs)


class MDMGenerationMixin:
    """
    Mixin class for Masked Diffusion Model generation, adapted from the **Dream** model's generation utils.
    """

    @torch.no_grad()
    def diffusion_generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[MDMGenerationConfig] = None,
        **kwargs,
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        # 1. Prepare the generation config.
        generation_config = self._mdm_prepare_generation_config(generation_config, **kwargs)

        if input_ids is None:
            raise ValueError("`input_ids` must be provided for diffusion generation.")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = input_ids.shape[-1] + generation_config.max_new_tokens
        elif generation_config.max_length is None:
            generation_config.max_length = input_ids.shape[-1]
        if generation_config.max_length < input_ids.shape[-1]:
            raise ValueError(f"`max_length` ({generation_config.max_length}) must be >= input length ({input_ids.shape[-1]}).")
        if generation_config.steps <= 0:
            raise ValueError(f"`steps` must be > 0, got {generation_config.steps}")
        if not (0.0 < generation_config.eps <= 1.0):
            raise ValueError(f"`eps` must be in (0, 1], got {generation_config.eps}")
        
        # 3. Expand the input for multi-sequence generation.
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(f"`attention_mask` must be 2D [batch, seq], got shape={tuple(attention_mask.shape)}")
            if attention_mask.shape != input_ids.shape:
                raise ValueError(f"`attention_mask` shape {tuple(attention_mask.shape)} must match `input_ids` shape {tuple(input_ids.shape)} before canvas padding.")
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)

        if generation_config.mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")
        
        pad_len = generation_config.max_length - input_ids.shape[-1]
        input_ids = F.pad(input_ids, (0, pad_len), value=generation_config.mask_token_id)
        if attention_mask is not None and pad_len > 0:
            # New canvas region corresponds to generation region; keep it valid by default.
            attention_mask = F.pad(attention_mask, (0, pad_len), value=1)

        # 4. Run the sampling loop.
        return self._mdm_sample(
            x=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )


    def _mdm_prepare_generation_config(
        self, 
        generation_config: Optional[GenerationConfig], 
        **kwargs
    ) -> MDMGenerationConfig:
        # Build the MDM generation config from the incoming generation_config.
        if generation_config is None:
            generation_config = self.generation_config
        
        # Use MDMGenerationConfig as the target class.
        if not isinstance(generation_config, MDMGenerationConfig):
            generation_config = MDMGenerationConfig.from_dict(generation_config.to_dict())
        else:
            generation_config = copy.deepcopy(generation_config)

        # Update kwargs.
        generation_config.update(**kwargs)
        return generation_config

    
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask


    def _mdm_sample(
        self,
        x: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MDMGenerationConfig
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        
        # Extract params from config

        mask_token_id = generation_config.mask_token_id

        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if generation_config.output_history else None

        # Fixed tokens = input context (should never be remasked in p2)
        fix_mask = (x != mask_token_id)

        pad_attention_mode = generation_config.pad_attention_mode
        if pad_attention_mode not in ("attend_pad", "mask_pad"):
            raise ValueError(f"`pad_attention_mode` must be one of ['attend_pad', 'mask_pad'], got {pad_attention_mode}")
        use_custom_attention_mask = attention_mask is not None
        if use_custom_attention_mask:
            if attention_mask.ndim != 2:
                raise ValueError(f"`attention_mask` must be 2D [batch, seq], got shape={tuple(attention_mask.shape)}")
            if attention_mask.shape != x.shape:
                raise ValueError(f"`attention_mask` shape {tuple(attention_mask.shape)} must match generation canvas shape {tuple(x.shape)}.")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.long)

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        for i in range(steps):
            mask_index = (x == mask_token_id)
            if not mask_index.any(): # Stop if no tokens are masked
                break

            if use_custom_attention_mask:
                # Caller-provided mask has highest priority.
                gen_attention_mask = attention_mask
            else:
                # attend_pad lets PAD positions participate in attention;
                # mask_pad excludes them using the configured PAD token ID.
                if pad_attention_mode == "attend_pad":
                    gen_attention_mask = torch.ones_like(x, dtype=torch.long, device=x.device)
                else:
                    if generation_config.pad_token_id is None:
                        gen_attention_mask = None
                    else:
                        gen_attention_mask = (x != generation_config.pad_token_id).long()

            # Bidirectional attention is used throughout denoising.
            outputs = self(input_ids=x, attention_mask=gen_attention_mask, is_causal=False)
            logits = outputs.logits

            # Optional: block PAD sampling if requested.
            if generation_config.pad_token_id is not None and generation_config.block_pad_token:
                logits[..., generation_config.pad_token_id] = float('-inf')
            logits = restrict_logits_to_allowed_tokens(
                logits, generation_config.allowed_token_ids
            )
                
            mask_logits = logits[mask_index] # select the masked positions
            t = timesteps[i]
            s = timesteps[i + 1]

            # Start sampling.
            if alg == "origin":
                # s / t is the fraction of masked positions that stay masked.
                # 1 - s / t is the probability that a masked token is revealed in this step, meaning replaced by a freshly sampled token.
                # s / t falls as i grows, so p_transfer rises and later steps reveal more tokens at once.
                p_transfer = 1 - s / t if i < steps - 1 else 1 # at i == steps - 1 every remaining masked token is revealed
                # Tensor of mask_token_id with one entry per masked token.
                x0 = torch.full_like(x[mask_index], fill_value=mask_token_id, device=x.device, dtype=torch.long)
                # Random tensor shaped like x0, recording which positions fall below p_transfer.
                transfer_index_t_s = torch.rand(*x0.shape, device=x.device) < p_transfer
                # Sample the predicted tokens and write them into the matching entries of x0.
                if transfer_index_t_s.any():
                    _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                    x0[transfer_index_t_s] = sampled_tokens
                # Write x0 back into x, which holds the user prompt and the denoised output.
                x[mask_index] = x0

            elif alg == "p2":
                # Compute confidence and sampled tokens for the entire sequence:
                #   conf_full: [B, L], confidence of the sampled token at each position
                #   x0_full:  [B, L], sampled token IDs for each position
                conf_full, x0_full = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)

                # Construct full_conf matrix and mask out fixed positions
                # Only positions in (~fix_mask) are candidates for masking/unmasking
                full_conf = conf_full.clone()
                full_conf[fix_mask] = float("inf")
                # Prevent NaNs or extreme values from interfering
                full_conf = torch.where(torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf")))

                # Increase the fraction of unmasked positions linearly over the steps.
                kappa_t = (i + 1) / steps
                # Calculate how many positions to re-mask per sample
                # = number of variable positions * (1 - kappa_t)
                num_positions = (~fix_mask).sum(dim=1)  # [B]
                num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                # Boundaries: at least 0, at most total number of variable positions
                num_to_mask = num_to_mask.clamp_min(0)
                num_to_mask = torch.minimum(num_to_mask, num_positions)

                # Select the lowest-confidence positions within (~fix_mask) for re-masking
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)  # [B, L]
                max_k = int(num_to_mask.max().item())
                if max_k > 0:
                    topk_idx = sorted_idx[:, :max_k]  # [B, max_k]
                    row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)  # [B, max_k]

                    to_mask = torch.zeros_like(x, dtype=torch.bool)
                    batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)  # [B, max_k]
                    valid_batch = batch_arange[row_mask]  # [sum_k]
                    valid_col   = topk_idx[row_mask]      # [sum_k]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x, dtype=torch.bool)

                # Apply re-masking: set selected positions back to mask_token_id
                x[to_mask] = mask_token_id

                # For positions that started as mask and were not re-masked, unmask them with sampled tokens
                keep_unmask = mask_index & (~to_mask)
                x[keep_unmask] = x0_full[keep_unmask]
                


            elif alg in ["maskgit_plus", "entropy", "topk_margin"]:
                # Confidence-based sampling (maskgit, entropy, etc.)
                
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                confidence = confidence.to(mask_logits.dtype)

                # Calculate number of mask tokens per sample
                num_mask_tokens_per_sample = mask_index.sum(dim=1)  # [batch_size]
                
                # Calculate transfer tokens per sample
                if i < steps - 1:
                    number_transfer_tokens_per_sample = (num_mask_tokens_per_sample.float() * (1 - s / t)).long()
                else:
                    number_transfer_tokens_per_sample = num_mask_tokens_per_sample
                number_transfer_tokens_per_sample = torch.minimum(number_transfer_tokens_per_sample, num_mask_tokens_per_sample)
                
                # Build full confidence matrix
                full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                # Get maximum transfer tokens for efficient batching
                max_transfer_tokens = number_transfer_tokens_per_sample.max().item()
                
                if max_transfer_tokens > 0:
                    if alg_temp is None or alg_temp <= 0:
                        # Use topk for each sample
                        _, all_transfer_indices = torch.topk(full_confidence, max_transfer_tokens, dim=1)  # [batch_size, max_transfer_tokens]
                    else:
                        # Robust vectorized sampling via Gumbel-TopK (no replacement)
                        # Handles rows with fewer valid positions than requested and rows with no valid positions
                        # full_confidence has -inf for invalid positions; keep them -inf so they won't be selected
                        scaled_logits = full_confidence / alg_temp
                        # Uniform in (0,1) to avoid log(0)
                        uniform = torch.rand_like(scaled_logits).clamp_(min=1e-20, max=1 - 1e-20)
                        gumbel_noise = -torch.log(-torch.log(uniform))
                        scores = scaled_logits + gumbel_noise
                        _, all_transfer_indices = torch.topk(scores, max_transfer_tokens, dim=1)  # [batch_size, max_transfer_tokens]
                    
                    # Create mask for valid transfers (handle variable number of transfers per sample)
                    batch_size = x.size(0)
                    valid_mask = torch.arange(max_transfer_tokens, device=x.device).unsqueeze(0) < number_transfer_tokens_per_sample.unsqueeze(1)  # [batch_size, max_transfer_tokens]
                    
                    # Get valid transfer indices and corresponding batch indices
                    valid_transfer_indices = all_transfer_indices[valid_mask]  # [total_valid_transfers]
                    valid_batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(all_transfer_indices)[valid_mask]  # [total_valid_transfers]
                    
                    # Prepare the transfer data
                    x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    
                    # Batch update using advanced indexing
                    x[valid_batch_indices, valid_transfer_indices] = x_[valid_batch_indices, valid_transfer_indices]
            
            else:
                raise NotImplementedError(f"Algorithm {alg} not implemented.")

            if histories is not None:
                histories.append(x.clone())

        if generation_config.return_dict_in_generate:
            return MDMModelOutput(sequences=x, history=None if histories is None else tuple(histories))
        else:
            return x
