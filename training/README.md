# PhaGen training with VeOmni

**PhaGen was trained using the [VeOmni training framework](https://github.com/ByteDance-Seed/VeOmni).**
The `veomni` modules provide distributed training, dataset loading and
packing, optimization, learning-rate scheduling and checkpoint management. The
PhaGen modules implement the hierarchical DNA model, masked-diffusion
training and validation.

VeOmni is developed by ByteDance and distributed under Apache-2.0. Original
copyright and license headers are retained in the source. See the repository's
[LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md) for attribution and adapted
upstream components.

## Layout

```text
veomni/                      VeOmni framework and PhaGen training model
tasks/train_torch_dllm.py      Masked-diffusion training and validation loop
configs/paper_reference.yaml  Paper training configuration
scripts/split_parquet.py      Seeded Parquet train/validation splitter
train.sh                     Portable torchrun launcher
tests/                       Small CPU checks of training behavior
```

The model modules are `veomni.models.transformers.phagen`, with classes
`PhaGenConfig` and `PhaGenModel` and `model_type="phagen"`. The paper configuration
uses the `adamw` optimizer; `anyprecision_adamw` is also available.

## Install

Install the training environment separately from the lightweight inference package.
From the repository root, using Python 3.10+ and a CUDA-enabled PyTorch 2.5 installation:

```bash
python -m pip install -e './training[test]'
```

The package installs the `veomni` modules. Training uses Transformers
4.54.1, TorchData, Datasets and ByteCheckpoint; other framework components
may require additional optional dependencies from VeOmni. Do not install a second
VeOmni checkout in the same environment.

The paper configuration requests `flash_attention_2`, which requires a compatible
`flash-attn` installation. The optional `cuda` extra includes FlashAttention and
Liger Kernel; match these packages to your PyTorch/CUDA build. For a portable
alternative, pass `--model.attn_implementation sdpa` to the launcher.

Training datasets, checkpoints, experiment logs and caches are external inputs
and outputs.

## Configuration and launch

`configs/paper_reference.yaml` provides the paper training hyperparameters:

| Setting | Paper run |
| --- | --- |
| Token canvas | 131,072 |
| Epochs | 50 |
| Parallelism | DDP, 4 training ranks |
| Batch size | Global 72, micro 1; 18 accumulation steps with 4 ranks |
| Optimizer | AdamW, weight decay 0.1, max gradient norm 1.0 |
| Learning rate | 2e-5, cosine decay to 3e-6, no warmup |
| Seed | 42 |
| Precision | FP32 parameters with BF16 autocast |
| Gradient checkpointing / offload | Disabled |
| Checkpoint / validation interval | 200 steps |
| Validation anchor seed | 114514 |

Replace the model and data paths, and choose a new output directory:

```bash
NPROC_PER_NODE=4 bash training/train.sh \
  --model.model_path /path/to/initialization_checkpoint \
  --data.train_path /path/to/train.parquet \
  --data.val_path /path/to/validation.parquet \
  --train.output_dir /path/to/new_run
```

The launcher defaults to the paper reference configuration. Use
`CONFIG=/path/to/config.yaml` for another configuration. It respects externally
set `CUDA_VISIBLE_DEVICES` and supports `NNODES`, `NODE_RANK`, `MASTER_ADDR` and
`MASTER_PORT`. Relative paths inside the configuration resolve from `training/`.
By default, the launcher starts a new run with W&B logging disabled. Set
`--train.use_wandb true` to enable experiment logging.

The initialization weights must be a local Hugging Face checkpoint with model
configuration, tokenizer and weights. Use the public PhaGen configuration
(`model_type="phagen"`, `architectures=["PhaGenModel"]`) with the original model
dimensions and token IDs. The paper run initialized from a preceding run's
step-2,400 checkpoint, which is currently unavailable.
Training with another initialization is a new experiment; exact
retraining to the released paper weights is not established by this reference.

## Training behavior

- Complete BOS/EOS-delimited sequences are packed into fixed token canvases.
  Packed PAD positions participate in attention.
- Mask ratios are sampled continuously and clamped to `[0.002, 0.998]`.
  BOS/EOS are always masked; PAD can be masked and predicted.
- Loss is the mean cross-entropy over supervised masked positions without
  timestep reweighting.
- Validation includes fixed-anchor and random-once masked losses.

`veomni/optim/optimizer.py` retains the upstream `AnyPrecisionAdamW` implementation
and its Meta / Llama 2 license header. The license terms are provided in
[licenses/LLAMA2_LICENSE.txt](licenses/LLAMA2_LICENSE.txt); see [NOTICE.md](NOTICE.md)
for attribution.

## Prepare a train/validation split

`scripts/split_parquet.py` creates a seed-based Bernoulli row split.
It expects a Parquet input with the DNA column `sequence`; other columns are
preserved. Keep the input row order fixed for reproducible splits:

```bash
python training/scripts/split_parquet.py \
  --input /path/to/megadna_training_dataset.parquet \
  --val-ratio 0.05 --seed 42 \
  --train-output /path/to/train.parquet \
  --val-output /path/to/validation.parquet
```

The paper split has 93,683 training records and 4,982 validation records. Record
IDs are disjoint, but 549 identical DNA sequence contents occur in both partitions,
affecting 553 validation records (11.10%). The splitter does not deduplicate or
alter DNA sequences.

## Lightweight checks

```bash
bash -n training/train.sh
bash training/train.sh --help
cd training
python -m pytest
```

The CPU tests use synthetic packed sequences and a small randomly initialized
model. They check masking, PAD behavior, the unweighted objective and a
forward/backward optimizer step. They were run with Python 3.10, PyTorch 2.5.0 and
Transformers 4.54.1; this is not a complete specification of the original paper
runtime. The checks do not cover full GPU training or distributed resume, which
require validation with the intended data and hardware.
