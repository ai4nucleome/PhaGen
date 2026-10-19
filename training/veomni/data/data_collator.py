# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ..distributed.parallel_state import get_parallel_state
from ..utils.seqlen_pos_transform_utils import len2culen, pos2culen
from .constants import IGNORE_INDEX


@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...


class CollatePipeline:
    def __init__(self, data_collators: Optional[Union[Callable, List[Callable]]] = None):
        """
        Args:
            data_collators: a list of data collators or a single data collator
        """

        if not isinstance(data_collators, (list, tuple)):
            data_collators = [data_collators]
        self.data_collators = data_collators

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        """
        process data batch through data collators.

        Args:
            batch: the original input data batch

        Returns:
            batch: the processed data batch

        """
        for data_collator in self.data_collators:
            batch = data_collator(batch)
        return batch


@dataclass
class DataCollatorWithPadding(DataCollator):
    """
    Data collator with padding.
    """

    pad_token_id: int = 0

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            # process padding features
            if key in ["input_ids", "attention_mask", "position_ids", "images_seq_mask"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            elif key in ["labels", "labels_image"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])

        return batch

@dataclass
class DataCollatorWithPaddingMasking(DataCollator):
    """
    Data collator with padding and masking.
    """
    def __init__(self, mask_token_id: int, pad_token_id: int, bos_token_id: int, eos_token_id: int):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _random_masking(self, input_ids: "torch.Tensor",) -> "torch.Tensor":
        """
        Randomly mask input_ids.
        """
        mask_ratio = torch.rand(1, device=input_ids.device).clamp(1/500, 1-1/500) # [0.002,0.998]
        mask_indices = torch.rand_like(input_ids.float()) < mask_ratio # 生成一个与输入id等长度的均匀分布的tensor，每个元素都在[0,1]之间；将小于mask_ratio的元素作为mask的地方 
        
        structural_mask = (input_ids == self.bos_token_id) | (input_ids == self.eos_token_id)
        mask_indices = mask_indices | structural_mask

        input_ids[mask_indices] = self.mask_token_id # 将小于mask_ratio大小的位置的元素（即为true的地方）设置为mask token的id
        return input_ids, mask_ratio.repeat(input_ids.size(0))
    
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                if key == "input_ids":
                    feature[key], mask_ratio = self._random_masking(feature[key])
                    batch["mask_ratio"].append(mask_ratio)
                batch[key].append(feature[key])

        # 记录 mask_ratio 每个样本的原始长度和值
        mask_ratio_lengths = None
        mask_ratio_values = None
        if "mask_ratio" in batch:
            mask_ratio_lengths = [len(item) for item in batch["mask_ratio"]]
            mask_ratio_values = [item[0] for item in batch["mask_ratio"]]  # 每个样本的 mask_ratio 值
        
        for key in batch.keys():
            # process padding features
            if key in ["input_ids", "attention_mask", "position_ids", "images_seq_mask", "mask_ratio", "input_bak"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.pad_token_id if key == "input_ids" else 0)
            elif key in ["labels", "labels_image"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])
        
        # 将 mask_ratio 中 padding 的 0 替换为对应样本的 mask_ratio 值
        if "mask_ratio" in batch:
            for i, (length, value) in enumerate(zip(mask_ratio_lengths, mask_ratio_values)):
                batch["mask_ratio"][i, length:] = value

        # 在label中把没有被 mask 的token的label设置成-100，也就是ignore_index；这样在计算loss时可以忽略他们
        if "labels" in batch:
            batch["labels"][batch["input_ids"] != self.mask_token_id] = IGNORE_INDEX
            
        return batch


@dataclass
class DataCollatorWithPacking(DataCollator):
    """
    Data collator with packing.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        seqlens = torch.tensor([len(feature["input_ids"]) for feature in features], dtype=torch.long)
        batch = {"cu_seqlens": len2culen(seqlens)}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features])
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        return batch

@dataclass
class DataCollatorWithPackingMasking(DataCollator):
    """
    Data collator with packing.
    """
    def __init__(self, mask_token_id: int,):
        self.mask_token_id = mask_token_id
        
    def _random_masking(self, input_ids: "torch.Tensor",) -> "torch.Tensor":
        """
        Randomly mask input_ids.
        """
        mask_ratio = torch.rand(1, device=input_ids.device).clamp(1/500, 1-1/500) # [0.002, 0.998]
        mask_indices = torch.rand_like(input_ids.float()) < mask_ratio # 生成一个与输入id等长度的均匀分布的tensor，每个元素都在[0,1]之间；将小于mask_ratio的元素作为mask的地方 
        input_ids[mask_indices] = self.mask_token_id # 将小于mask_ratio大小的位置的元素（即为true的地方）设置为mask token的id
        return input_ids, mask_ratio.repeat(input_ids.size(0))

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        seqlens = torch.tensor([len(feature["input_ids"]) for feature in features], dtype=torch.long)
        batch = {"cu_seqlens": len2culen(seqlens)}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features])
                # Apply random masking only to input_ids, concatenate all features
                if input_name == "input_ids":
                    # 对每个sample的input_ids进行随机mask，返回mask后的input_ids和mask_ratio
                    masked_features = [self._random_masking(feature["input_ids"]) for feature in features]
                    # 将所有sample的mask后的input_ids拼接起来，并添加一个维度，使得每个sample的input_ids都在一个序列中
                    batch[input_name] = torch.cat(list(i[0] for i in masked_features))
                    # 将所有sample的mask_ratio拼接起来，并添加一个维度，使得每个sample的mask_ratio都在一个序列中
                    batch["mask_ratio"] = torch.cat(list(i[1] for i in masked_features))
                else:
                    batch[input_name] = torch.cat([feature[input_name] for feature in features])
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])
                
        # 在label中把没有被 mask 的token的label设置成-100，也就是ignore_index；这样在计算loss时可以忽略他们
        if "labels" in batch:
            batch["labels"][batch["input_ids"] != self.mask_token_id] = IGNORE_INDEX
        return batch

@dataclass
class DataCollatorWithPositionIDs(DataCollator):
    """
    Data collator with packing by position ids.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = {}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(feature["input_ids"])) for feature in features]
            ).unsqueeze(0)

        if "labels" in batch:
            cu_seqlens = pos2culen(batch["position_ids"])
            batch["labels"][:, cu_seqlens[1:-1]] = IGNORE_INDEX

        return batch


@dataclass
class DataCollatorWithPositionIDsMasking(DataCollator):
    """
    Data collator with masking.
    """
    def __init__(self, mask_token_id: int,):
        self.mask_token_id = mask_token_id
        
    def _random_masking(self, input_ids: "torch.Tensor",) -> "torch.Tensor":
        """
        Randomly mask input_ids.
        """
        mask_ratio = torch.rand(1, device=input_ids.device).clamp(1/500, 1-1/500) # [0.002, 0.998]
        mask_indices = torch.rand_like(input_ids.float()) < mask_ratio # 生成一个与输入id等长度的均匀分布的tensor，每个元素都在[0,1]之间；将小于mask_ratio的元素作为mask的地方 
        input_ids[mask_indices] = self.mask_token_id # 将小于mask_ratio大小的位置的元素（即为true的地方）设置为mask token的id
        return input_ids, mask_ratio.repeat(input_ids.size(0))

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        # 对一个 microbatch 的数据进行处理
        batch = {}
        for input_name in features[0].keys():
            # "input_ids", "attention_mask", "labels", "position_ids"
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids"):
                # Apply random masking only to input_ids, concatenate all features
                if input_name == "input_ids":
                    # 对每个sample的input_ids进行随机mask，返回mask后的input_ids和mask_ratio
                    masked_features = [self._random_masking(feature[input_name]) for feature in features]
                    
                    # Concatenate masked samples; position IDs retain sample boundaries.
                    # 将所有sample的mask后的input_ids拼接起来，并添加一个维度，使得每个sample的input_ids都在一个序列中
                    batch[input_name] = torch.cat(list(i[0] for i in masked_features), dim=-1).unsqueeze(0)
                    # 将所有sample的mask_ratio拼接起来，并添加一个维度，使得每个sample的mask_ratio都在一个序列中
                    batch["mask_ratio"] = torch.cat(list(i[1] for i in masked_features), dim=-1).unsqueeze(0)
                else:
                    batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        # 给每个样本生成其token的postion id，并concat起来
        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(feature["input_ids"])) for feature in features]
            ).unsqueeze(0)

        # 在label中把没有被 mask 的token的label设置成-100，也就是ignore_index；这样在计算loss时可以忽略他们
        if "labels" in batch:
            batch["labels"][batch["input_ids"] != self.mask_token_id] = IGNORE_INDEX
        return batch
@dataclass
class NoopDataCollator(DataCollator):
    """
    Data collator with no operation, used in dynamic batch dataloader at main process.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        return features


@dataclass
class UnpackDataCollator(DataCollator):
    """
    Data collator to unpack examples, used in dynamic batch dataloader at worker process.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        # Unwrap the batch produced by the worker data loader.
        return features[0]


@dataclass
class MakeMicroBatchCollator(DataCollator):
    """
    Data collator to build micro batches, used in mapping dataloader.
    """

    num_micro_batch: int
    internal_data_collator: "DataCollator"

    def __call__(self, features: Sequence[Tuple[Dict[str, "torch.Tensor"]]]) -> List[Dict[str, "torch.Tensor"]]:
        # features 是gradient accumulation时一个GPU进程要前向传播的sample的数目
        # num_micro_batch 是gradinet accumulation的step数目
        features = list(features)
        if len(features) == 0:
            return []

        # Use a micro batch of at least one sample for a short final batch.
        micro_batch_size = len(features) // self.num_micro_batch
        if micro_batch_size == 0:
            micro_batch_size = 1

        for i in range(len(features)):
            # 只取了每个sample的第一个chunk, 因为每个sample可能被切分为多个chunks
            features[i] = features[i][0]  # 1-to-N inverse transform

        micro_batches = []
        for i in range(0, len(features), micro_batch_size):
            # 使用 collate_fn 来处理每个batch的数据，这里使用的是DataCollatorWithPaddingMasking
            micro_batches.append(self.internal_data_collator(features[i : i + micro_batch_size]))

        return micro_batches


@dataclass
class TextSequenceShardCollator(DataCollator):
    """
    Data collator to chunk inputs according to sequence parallelism.
    Args:
        rmpad: whether the samples is packing or not.
        rmpad_with_pos_ids: whether the samples is packing by position ids or not.
        pad_token_id: the id of the padding token.
    """

    rmpad: bool
    rmpad_with_pos_ids: bool
    pad_token_id: int = 0

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

    def sp_slice(self, tensor: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
        """
        Slices a tensor along the specified dimension for sequence parallelism.
        """
        seq_length = tensor.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return tensor.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_length: int = 0
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        if pad_length == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        input_ids = batch.pop("input_ids")
        labels = batch.pop("labels")[..., 1:].contiguous()  # shift labels
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        if self.rmpad_with_pos_ids:  # mask the last token of each sequence
            cu_seqlens = pos2culen(batch["position_ids"])
            labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX
        elif self.rmpad:
            labels = labels.view(-1)
            labels[batch["cu_seqlens"][1:-1] - 1] = IGNORE_INDEX
        else:
            if "position_ids" not in batch:  # we should calculate the position ids before chunking
                batch["position_ids"] = torch.arange(0, input_ids.size(-1)).unsqueeze(0)

        # sp padding
        seq_length = input_ids.size(-1)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        pad_length = sp_chunk_size * self.sp_size - seq_length

        input_ids = self.sp_padding(input_ids, dim=-1, pad_value=self.pad_token_id, pad_length=pad_length)
        labels = self.sp_padding(labels, dim=-1, pad_value=IGNORE_INDEX, pad_length=pad_length)

        if self.rmpad_with_pos_ids:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=1, pad_length=pad_length
            )
        else:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=0, pad_length=pad_length
            )

        if self.rmpad:
            if pad_length > 0:
                batch["cu_seqlens"] = F.pad(
                    batch["cu_seqlens"], (0, 1), "constant", batch["cu_seqlens"][-1].item() + pad_length
                )
        else:
            batch["position_ids"] = self.sp_padding(batch["position_ids"], dim=-1, pad_value=0, pad_length=pad_length)

        # sp slice
        batch["input_ids"] = self.sp_slice(input_ids, dim=-1)
        batch["labels"] = self.sp_slice(labels, dim=-1)

        return batch
