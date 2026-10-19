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


import torch
import torch.nn.functional as F


def len2culen(seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the sequence lengths to cumulative sequence lengths.

    NOTE: flash attention only accepts int32 cu_seqlens.
    """
    return F.pad(torch.cumsum(seqlens, dim=0), (1, 0)).type(torch.int32) # 输入 [800, 600, 800] 返回 [0, 800, 1400, 2200]


def culen2len(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the cumulative sequence lengths to sequence lengths.
    """
    return cu_seqlens.diff() # 输入 [0, 800, 1400, 2200] 返回 [800, 600, 800] 即返回每段序列的长度 (diff 即相邻两个元素的差值)


def pos2culen(position_ids: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the position ids to cumulative sequence lengths.
    """
    if position_ids.dim() == 3:  # (batch_size, dim, seq_length):
        position_ids = position_ids[:, 0, :]

    position_ids = position_ids.flatten() # 把 postion id 展开成一维向量
    indices_q = torch.arange(position_ids.size(0), dtype=torch.int32, device=position_ids.device) # 获取position id长度的list
    # indices_q[position_ids == 0]，把 position id 转成 cu_seqlens, 再使用 F.pad 在序列结尾 pad 上序列的总长度作为最后一个序列的 cu_seqlens
    return F.pad(indices_q[position_ids == 0], (0, 1), "constant", position_ids.size(0))


def culen2pos(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
    """
    Converts the cumulative sequence lengths to position ids.
    """
    seqlens = culen2len(cu_seqlens).cpu()
    position_ids = torch.cat([torch.arange(length, dtype=torch.long, device=cu_seqlens.device) for length in seqlens])
    return position_ids.unsqueeze(0)
