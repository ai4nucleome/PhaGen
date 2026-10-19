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


import os
import random
from sortedcontainers import SortedList
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Union

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.dist_utils import main_process_first


logger = logging.get_logger(__name__)


class DummyDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        ]


class MappingDataset(Dataset):
    """
    Mapping dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]


class IterativeDataset(IterableDataset):
    """
    Iterative dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                yield self._transform(sample)
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)


class PackingMappingDataset(Dataset):
    """
    Dynamic bin-packing dataset for sequence packing.

    Pre-tokenizes all sequences once, then uses greedy bin-packing to group
    complete sequences into fixed-size chunks of max_seq_len. Each chunk contains
    multiple complete sequences delimited by [BOS]/[EOS], padded with [PAD].

    Format per chunk:
        [BOS] seq1 [EOS] [BOS] seq2 [EOS] ... [PAD] [PAD] ... [PAD]

    Calling set_epoch(epoch) re-shuffles and re-packs with a different seed,
    so the model sees different sequence combinations each epoch.
    """

    def __init__(
        self,
        all_token_ids: List[List[int]],
        max_seq_len: int,
        pad_token_id: int,
        seed: int = 0,
    ):
        """
        Args:
            all_token_ids: Pre-tokenized sequences, each as [BOS, ...tokens..., EOS]
            max_seq_len: Fixed chunk size (e.g., 131072)
            pad_token_id: Token id used for padding
            seed: Initial random seed for packing
        """
        self.all_token_ids = all_token_ids
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Pre-compute invariants (avoid recomputing each epoch)
        self._seq_lens = [len(t) for t in all_token_ids]
        self._total_real_tokens = sum(self._seq_lens)
        logger.info(
            f"[PackingMappingDataset] {len(all_token_ids)} sequences, "
            f"min={min(self._seq_lens)}, max={max(self._seq_lens)}, "
            f"mean={self._total_real_tokens/len(self._seq_lens):.0f}"
        )

        # Check no sequence exceeds max_seq_len
        oversized = [l for l in self._seq_lens if l > max_seq_len]
        if oversized:
            raise ValueError(
                f"{len(oversized)} sequences exceed max_seq_len={max_seq_len}. "
                f"Max found: {max(oversized)}. Cannot pack without truncation."
            )

        self._pack(seed)

    def _pack(self, seed: int):
        """Best Fit (BF) bin-packing with per-epoch shuffling.

        1. Shuffle indices with seed (different seed → different packing each epoch)
        2. For each sequence, find the bin with LEAST remaining space that still fits
        3. Uses SortedList for O(N √B) total complexity

        No length sorting — the shuffle order directly determines packing combinations,
        giving maximum per-epoch diversity while best-fit search maintains high efficiency.
        """

        rng = random.Random(seed)
        indices = list(range(len(self.all_token_ids)))
        rng.shuffle(indices)

        # SortedList of (remaining_cap, bin_id) — sorted by cap ascending
        # All operations (add/pop/bisect) are O(log B) amortized
        caps = SortedList()
        # Each bin stores a list of sequence indices (not the actual tokens)
        bin_indices: List[List[int]] = []

        for idx in indices:
            seq_len = self._seq_lens[idx]

            # Binary search: find smallest remaining_cap >= seq_len (best fit)
            # Tuple comparison: (seq_len, -1) ensures we find first cap >= seq_len
            pos = caps.bisect_left((seq_len, -1))

            if pos < len(caps):
                # Found a bin — the one at 'pos' has the tightest fit
                old_cap, bin_id = caps.pop(pos)
                bin_indices[bin_id].append(idx)
                new_cap = old_cap - seq_len
                caps.add((new_cap, bin_id))
            else:
                # No bin fits, open a new one
                bin_id = len(bin_indices)
                bin_indices.append([idx])
                new_cap = self.max_seq_len - seq_len
                caps.add((new_cap, bin_id))

        # Store sequence index lists (lazy: tensors constructed in __getitem__)
        self._bin_seq_indices = bin_indices

        # Shuffle chunk order for training diversity
        self._chunk_order = list(range(len(bin_indices)))
        rng.shuffle(self._chunk_order)

        # Stats (use pre-computed _total_real_tokens)
        total_slots = len(bin_indices) * self.max_seq_len
        efficiency = self._total_real_tokens / total_slots * 100 if total_slots > 0 else 0
        logger.info(
            f"[PackingMappingDataset] BF packed into {len(bin_indices)} chunks, "
            f"efficiency: {efficiency:.1f}%, seed: {seed}"
        )

    def _finalize_chunk(self, seq_indices: List[int]) -> Dict[str, "torch.Tensor"]:
        """Assemble sequences by index, pad, and create tensors.

        - attention_mask: all 1s (PAD participates in attention for diffusion training)
        - labels: PAD positions use pad_token_id (model learns to predict PAD after EOS)
        """
        # Pre-allocate tensor filled with pad_token_id
        input_ids = torch.full((self.max_seq_len,), self.pad_token_id, dtype=torch.long)
        offset = 0
        for idx in seq_indices:
            seq = self.all_token_ids[idx]
            seq_len = len(seq)
            input_ids[offset:offset + seq_len] = torch.as_tensor(seq, dtype=torch.long)
            offset += seq_len

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
            "labels": input_ids.clone(),  # PAD positions keep pad_token_id
        }

    def set_epoch(self, epoch: int):
        """Re-pack with a different seed each epoch for diverse combinations."""
        self._pack(seed=epoch)

    def __len__(self) -> int:
        return len(self._bin_seq_indices)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        # Lazy: construct tensor on-demand, not during packing
        seq_indices = self._bin_seq_indices[self._chunk_order[index]]
        return [self._finalize_chunk(seq_indices)]


def build_packing_dataset(
    data_path: str,
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "sequence",
    namespace: Literal["train", "test"] = "train",
    seed: int = 0,
) -> "PackingMappingDataset":
    """
    Build a packing dataset that groups complete sequences into fixed-size chunks.

    Args:
        data_path: Path to parquet/jsonl data files
        tokenizer: Tokenizer with bos_token_id, eos_token_id, pad_token_id
        max_seq_len: Fixed chunk size (e.g., 131072)
        text_keys: Column name(s) containing the sequence text
        namespace: Dataset split name
        seed: Initial random seed for packing

    Returns:
        PackingMappingDataset
    """
    # Load raw data
    data_files = []
    data_paths = data_path.split(",")
    for dp in data_paths:
        if os.path.isdir(dp):
            data_files.extend([os.path.join(dp, fn) for fn in os.listdir(dp)])
        elif os.path.isfile(dp):
            data_files.append(dp)
        else:
            raise FileNotFoundError(f"Dataset {dp} not exists.")

    file_ext = os.path.splitext(data_files[0])[-1][1:]
    if file_ext not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_ext} files are not supported.")
    file_ext = "json" if file_ext == "jsonl" else file_ext

    with main_process_first():
        dataset = load_dataset(file_ext, data_files=data_files, split=namespace)

    # Resolve text key
    if isinstance(text_keys, str):
        text_key = text_keys
    elif isinstance(text_keys, list):
        text_key = None
        for key in text_keys:
            if key in dataset.column_names:
                text_key = key
                break
        if text_key is None:
            raise ValueError(f"None of the keys {text_keys} found in dataset.")
    else:
        raise ValueError(f"text_keys must be str or list, got {type(text_keys)}")

    # Build bytes.translate table — map both upper and lower case (avoids upper() copy)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    unk_id = tokenizer.unk_token_id

    # Validate: all special token IDs must exist and fit in a byte (0..255)
    for name, tid in [("bos_token_id", bos_id), ("eos_token_id", eos_id), ("unk_token_id", unk_id)]:
        if tid is None:
            raise ValueError(f"tokenizer.{name} is None — tokenizer must define {name}.")
        if not (0 <= tid <= 255):
            raise ValueError(f"tokenizer.{name}={tid} is out of bytes range [0, 255].")

    table = bytearray([unk_id] * 256)
    for ch in 'ATCG':
        tok_id = tokenizer.convert_tokens_to_ids(ch)
        if tok_id is None:
            raise ValueError(f"Token '{ch}' not found in tokenizer vocabulary.")
        if not (0 <= tok_id <= 255):
            raise ValueError(f"Token '{ch}' has id={tok_id}, out of bytes range [0, 255].")
        table[ord(ch)] = tok_id
        table[ord(ch.lower())] = tok_id  # map lowercase too
    table = bytes(table)

    # Tokenize directly from dataset — no intermediate all_texts list, no upper()
    all_encoded = [None] * len(dataset)
    for i, ex in enumerate(dataset):
        text = ex[text_key]
        try:
            raw = text.encode('ascii')
        except UnicodeEncodeError as e:
            raise ValueError(
                f"Sequence {i} contains non-ASCII character at position {e.start}: "
                f"'{text[max(0,e.start-5):e.start+5]}...'. "
                f"DNA data should only contain ASCII characters (A/T/C/G)."
            ) from e
        body = list(raw.translate(table))
        all_encoded[i] = [bos_id] + body + [eos_id]

    logger.info(f"[build_packing_dataset] Tokenized {len(all_encoded)} sequences")

    return PackingMappingDataset(
        all_token_ids=all_encoded,
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        seed=seed,
    )


def build_dummy_dataset(size: int, max_seq_len: int) -> "Dataset":
    return DummyDataset(size=size, seq_length=max_seq_len)


def build_mapping_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
) -> "Dataset":
    """
    Build mapping dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
    Returns:
        Dataset: mapping dataset
    """
    # process the data path, whether it is one path or multiple paths
    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend(
                [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
            )
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    # obtain the file extension
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    with main_process_first():
        # a distributed synchronization context manager, default only the main process of each node will load data
        dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    return MappingDataset(data=dataset, transform=transform)


def build_iterative_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
) -> "IterableDataset":
    """ "
    Build iterative dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
    Returns:
        IterableDataset: iterative dataset
    """

    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend(
                [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
            )
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    parallel_state = get_parallel_state()
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    dataset = load_dataset(
        file_extenstion, data_files=data_files, split=namespace, streaming=True
    )
    dataset = dataset.shuffle(
        seed=seed, buffer_size=10_000
    )  # 先加载 10000 条数据到内存中，进行 shuffle；不是真正的随机打乱，受到 buffer size 大小的影响
    #! 由于数据是流式的，无法像mappingdataset那样预先给每个节点分配索引，因此，这里使用 split_dataset_by_node 来分配数据
    dataset = split_dataset_by_node(
        dataset, parallel_state.dp_rank, parallel_state.dp_size
    )

    return IterativeDataset(dataset, transform)
