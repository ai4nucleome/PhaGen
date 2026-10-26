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

def top_p_logits(logits, top_p=None):
    # top-p 采样
    ## 保留累积概率达到 top_p 的最小词汇集合；将其他词汇的概率设为负无穷
    
    # 1. 对logits进行排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # 2. 计算 softmax 概率，然后计算累积概率
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 3. 找出累积概率超过 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    # 4. 调整 mask，确保至少保留一个 token。
    #        这里相当于把累积概率向右移一位，来确保无论 top_p 设置为多少，始终能采样到 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # 5. 创建 mask 并应用到原始 logits 上
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    if top_k is None or top_k == 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    # torch.topk(logits, top_k)[0] 取出 topk 个元素的 value
    # torch.topk(logits, top_k)[0][..., -1, None] 取出 topk 个元素的 value 中的最后一个元素的数值
    # 然后通过 logits < torch.topk(logits, top_k)[0][..., -1, None] 找出小于第 topk 个元素 value 的位置
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    # 进行 mask
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, alg="origin"):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        # 根据概率随机选择
        x0 = dists.Categorical(probs=probs).sample()
    else:
        # 贪心采样
        _, x0 = probs.max(dim=-1)
    # 这里就是 softmax 之后的概率
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        # 计算概率最大和概率第二大的两个token之间的概率差异
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    elif alg == "entropy":
        # 信息熵，用来衡量一个概率分布的不确定性。
        # 值越小（越负），代表不确定性越低，即置信度越高
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
    history: Optional[Tuple[torch.FloatTensor]] = None

class MDMGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        # Set do_sample=True as default for MDM (since MDM handles its own sampling)
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = True
        
        super().__init__(**kwargs)
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.eps: float = kwargs.pop("eps", 1e-3) # 扩散过程结束的阈值
        self.steps: int = kwargs.pop("steps", 512) # 扩散步数
        self.alg: str = kwargs.pop("alg", 'entropy') # 采样算法
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", 0.0)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.mask_token_id = kwargs.pop("mask_token_id", None) # mask token id
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.pad_token_id = kwargs.pop("pad_token_id", None)


class MDMGenerationMixin:
    """
    Mixin class for Masked Diffusion Model generation, adapted from the Dream model's generation utils.
    """
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # 用于拓展批次大小，tensor.repeat_interleave(expand_size, dim=0)，按照 batch size 的维度进行拓展
        # torch.tensor([[1, 2, 3], [4, 5, 6]]).repeat_interleave(expand_size, dim=0)
        # tensor([[1, 2, 3],
        #         [1, 2, 3],
        #         [1, 2, 3],
        #         [4, 5, 6],
        #         [4, 5, 6],
        #         [4, 5, 6]])
        if expand_size == 1:
            return input_ids, attention_mask
        
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _mdm_prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs
    ) -> MDMGenerationConfig:
        # 从 generation_config 中获取配置，构建 MDM generation_config
        if generation_config is None:
            generation_config = self.generation_config
        
        # 使用 MDMGenerationConfig 作为目标类
        if not isinstance(generation_config, MDMGenerationConfig):
            generation_config = MDMGenerationConfig.from_dict(generation_config.to_dict())
        else:
            generation_config = copy.deepcopy(generation_config)

        # 更新 kwargs
        generation_config.update(**kwargs)
        return generation_config

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[MDMGenerationConfig] = None,
        **kwargs,
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        # 主要生成函数
        # 1. 准备生成配置
        generation_config = self._mdm_prepare_generation_config(generation_config, **kwargs)

        # 2. 准备输入
        input_ids = inputs
        attention_mask = kwargs.get("attention_mask", None)

        if input_ids is None:
            raise ValueError("`inputs` must be provided for diffusion generation.")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = input_ids.shape[-1] + generation_config.max_new_tokens
        
        # 3. 拓展输入用于多序列生成
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if generation_config.mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")
        
        input_ids = F.pad(input_ids, (0, generation_config.max_length - input_ids.shape[1]), value=generation_config.mask_token_id)
        attention_mask = None

        # 4. 运行采样循环
        return self._mdm_sample(
            x=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    def _mdm_sample(
        self,
        x: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MDMGenerationConfig
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        
        # Extract params from config

        max_length = generation_config.max_length
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
        # 找到输入序列的 token 的位置以及要生成的序列的token的位置
        fix_mask = (x != mask_token_id)

        # The model expects a bidirectional mask, so we just use the presence of pad_token_id
        # 这里是为了标记 padding token，即无意义的token，在生成过程计算双向注意力的时候需要忽略掉这些token
        gen_attention_mask = (x != generation_config.pad_token_id).long() if generation_config.pad_token_id is not None else None

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        for i in range(steps):
            mask_index = (x == mask_token_id)
            if not mask_index.any(): # Stop if no tokens are masked
                break

            # is_causal=False is crucial for bidirectional attention
            # 设置双向注意力，is_causal=False 是关键
            outputs = self(input_ids=x, attention_mask=gen_attention_mask, is_causal=False)
            # Reconstruct each masked token from the logits at the same position.
            logits = outputs.logits

            # Exclude PAD from the generated token distribution.
            if generation_config.pad_token_id is not None:
                logits[..., generation_config.pad_token_id] = float('-inf')
                
            mask_logits = logits[mask_index] # 进行掩码
            t = timesteps[i]
            s = timesteps[i + 1]

            # 开始采样
            if alg == "origin":
                # s / t: 这个比率代表了“保留遮盖”的程度。
                # 1 - s / t: 这就是本步中，每个被遮盖的 token 将被“揭示”（即被新采样的 token 替换）的概率。
                # The reveal probability increases as the remaining diffusion time decreases.
                p_transfer = 1 - s / t if i < steps - 1 else 1 # Reveal all remaining tokens at the final step.
                # 创建一个与被mask的token数量相同的tensor，填充为mask_token_id
                x0 = torch.full_like(x[mask_index], fill_value=mask_token_id, device=self.device, dtype=torch.long)
                # 创建一个与x0形状一致的概率tensor，记录当前位置的概率是否小于p_transfer
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                # 进行生成采样，得到预测出来的token，然后替换掉x0中对应的token
                _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                x0[transfer_index_t_s] = sampled_tokens
                # 将x0中的token替换掉x中对应的token，x 有两部分：用户的prompt和生成的去噪的结果
                x[mask_index] = x0

            elif alg == "p2":
                # Use sample_tokens to obtain confidence and candidate tokens for the whole sequence
                # kappa_t: fraction of tokens to keep unmasked (can be replaced with custom schedule)
                kappa_t = (i + 1) / steps

                # Compute confidence and sampled tokens for the entire sequence:
                #   conf_full: [B, L], confidence of the sampled token at each position
                #   x0_full:  [B, L], sampled token IDs for each position
                conf_full, x0_full = sample_tokens(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
                )

                # Construct full_conf matrix and mask out fixed positions
                # Only positions in (~fix_mask) are candidates for masking/unmasking
                full_conf = conf_full.clone()
                full_conf[fix_mask] = float("inf")
                # Prevent NaNs or extreme values from interfering
                full_conf = torch.where(
                    torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                )

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
                
                # Build full confidence matrix
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                # Get maximum transfer tokens for efficient batching
                max_transfer_tokens = number_transfer_tokens_per_sample.max().item()
                
                if max_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
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
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    
                    # Batch update using advanced indexing
                    x[valid_batch_indices, valid_transfer_indices] = x_[valid_batch_indices, valid_transfer_indices]
            
            else:
                raise NotImplementedError(f"Algorithm {alg} not implemented.")

            if histories is not None:
                histories.append(x.clone())

        if generation_config.return_dict_in_generate:
            return MDMModelOutput(sequences=x, history=histories)
        else:
            return x
