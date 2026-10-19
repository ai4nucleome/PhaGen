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


import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .batching_strategy import TextBatchingStrategy
from .data_collator import (
    CollatePipeline,
    DataCollatorWithPacking,
    DataCollatorWithPadding,
    DataCollatorWithPositionIDs,
    DataCollatorWithPaddingMasking,
    DataCollatorWithPackingMasking,
    DataCollatorWithPositionIDsMasking,
    MakeMicroBatchCollator,
    TextSequenceShardCollator,
    UnpackDataCollator,
)
from .dynamic_batching import DynamicBatchSizeDataLoader


if TYPE_CHECKING:
    from torch.utils.data import Dataset


logger = logging.get_logger(__name__)


class DistributedDataloader(StatefulDataLoader):
    dataset: "Dataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            # Packing repack may change dataset length across epochs.
            # Keep sampler internals in sync to avoid DistributedSampler length assertion failure.
            if hasattr(self.dataset, "__len__") and hasattr(self.sampler, "num_replicas"):
                dataset_len = len(self.dataset)
                num_replicas = self.sampler.num_replicas
                drop_last = getattr(self.sampler, "drop_last", False)
                if drop_last and dataset_len % num_replicas != 0:
                    num_samples = math.ceil((dataset_len - num_replicas) / num_replicas)
                else:
                    num_samples = math.ceil(dataset_len / num_replicas)
                self.sampler.num_samples = max(0, num_samples)
                self.sampler.total_size = self.sampler.num_samples * num_replicas
            self.sampler.set_epoch(epoch)


def build_dataloader(
    dataset: "Dataset",
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_batch_size: int,
    max_seq_len: int,
    train_steps: int,
    rmpad: bool = True,
    rmpad_with_pos_ids: bool = False,
    bsz_warmup_ratio: float = 0.02,
    bsz_warmup_init_mbtoken: int = 200,
    dyn_bsz_buffer_size: int = 500,
    dyn_bsz_margin: int = 0,
    collate_fn: Optional[Union[Callable, List[Callable]]] = None,
    num_workers: int = 8,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
    enable_masking: bool = False,
    mask_token_id: Optional[int] = None,
    padding_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> "DistributedDataloader":
    parallel_state = get_parallel_state()  # 获取并行的一些参数
    token_micro_bsz = (micro_batch_size * max_seq_len)  # 每个 GPU 在一次前向传播时所计算的最大 token 数目
    num_micro_batch = (global_batch_size // (  # 要进行几次前向传播，也就是 gradient accumulation 的次数
            micro_batch_size * parallel_state.dp_size)
    )  # num_micro_batch = num accumulation steps
    bsz_warmup_steps = int(train_steps * bsz_warmup_ratio)  # warmup 的步数
    use_rmpad = (rmpad or rmpad_with_pos_ids)  # 是否启用无填充训练/使用位置 ID 的无填充训练，rmpad 表示无填充训练，rmpad_with_pos_ids 表示使用位置 ID 的无填充训练
    logger.info_rank0(
        f"train_steps: {train_steps}, max_seq_len: {max_seq_len}, use_rmpad: {use_rmpad}, "
        f"bsz_warmup_steps: {bsz_warmup_steps}, bsz_warmup_init_mbtoken: {bsz_warmup_init_mbtoken}, "
        f"token_micro_bsz: {token_micro_bsz}, num_micro_batch: {num_micro_batch}, "
        f"micro_batch_size: {micro_batch_size}, global_batch_size: {global_batch_size}, "
        f"dp_size: {parallel_state.dp_size}, sp_size: {parallel_state.sp_size}."
    )

    if collate_fn is None:
        if enable_masking:  # Select a masking collator for diffusion training.
            if mask_token_id is None:
                raise ValueError("mask_token_id must be provided when enable_masking is True")
            if rmpad:
                collate_fn = DataCollatorWithPackingMasking(mask_token_id)
            elif rmpad_with_pos_ids:
                collate_fn = DataCollatorWithPositionIDsMasking(mask_token_id)  # 如果使用 rmpad_with_pos_ids时，使用这个 collate_fn 来进行 mask
            else:
                # Pass the tokenizer token IDs to the padding and masking collator.
                collate_fn = DataCollatorWithPaddingMasking(mask_token_id, padding_token_id, bos_token_id, eos_token_id)
        else:
            if rmpad:
                collate_fn = DataCollatorWithPacking()
            else:
                if rmpad_with_pos_ids:
                    collate_fn = DataCollatorWithPositionIDs()
                else:
                    collate_fn = DataCollatorWithPadding()

        if parallel_state.sp_enabled:
            collate_fn_list = [collate_fn]
            collate_fn_list.append(TextSequenceShardCollator(rmpad=rmpad, rmpad_with_pos_ids=rmpad_with_pos_ids))
            collate_fn = CollatePipeline(collate_fn_list)

    if isinstance(collate_fn, list):
        collate_fn = CollatePipeline(collate_fn)

    if use_rmpad:  # 如果使用了无填充的训练方式
        batching_strategy = TextBatchingStrategy(
            token_micro_bsz=token_micro_bsz - dyn_bsz_margin * max_seq_len,  # 原本计算的每个micro_batch的token减去动态batching的margin
            buffer_size=dyn_bsz_buffer_size,  # 默认是 200
            bsz_warmup_steps=(bsz_warmup_steps if bsz_warmup_steps else -1),  # 默认没有 warmup step
            bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,  # 默认是 200
        )
        dyn_bsz_collate_fn = collate_fn
        collate_fn = UnpackDataCollator()
    else:
        # 把 dataloader 返回的一个 batch 的数据按照 micro_batch_size 进行切分，然后返回一个 list，list 的每个元素是一个 batch 的数据
        collate_fn = MakeMicroBatchCollator(num_micro_batch=num_micro_batch, internal_data_collator=collate_fn)

    # 判断dataset是不是iterabledataset，如果不是则使用StatefulDistributedSampler
    # sampler的作用是：1.数据分发，决定每个GPU进程处理哪些数据；2.确保不同进程处理的数据是不同的；3.确保每个进程处理的数据量大致接近
    # 对于映射数据集 mappingDataset 需要使用 sampler；而对于 iterableDataset，由于在构建dataset时已经分片了，所以不需要使用 sampler
    # StatefulDistributedSampler的优势：
    # 支持断点续训
    # 记录当前epoch和step状态
    # 可以从任意位置恢复训练
    sampler = None
    if not isinstance(dataset, IterableDataset):
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=parallel_state.dp_size,  # 数据并行的进程数
            rank=parallel_state.dp_rank,  # 当前进程的 rank id
            shuffle=True,  # 是否打乱数据顺序
            seed=seed,  # 随机种子
        )

    dataloader = DistributedDataloader(
        dataset,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )
    if use_rmpad:
        dataloader = DynamicBatchSizeDataLoader(
            dataloader,
            batching_strategy=batching_strategy,
            collate_fn=dyn_bsz_collate_fn,
            num_micro_batch=num_micro_batch,
            length=train_steps,
            drop_last=drop_last,
        )

    return dataloader
