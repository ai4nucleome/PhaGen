# Attribution and license scope

PhaGen
Copyright 2026 The PhaGen authors

Licensed under the Apache License, Version 2.0 (see `LICENSE`).

## Third-party code

### `src/phagen/modeling_phagen.py` and the training PhaGenModel implementation

Adapts the MEGABYTE implementation by Phil Wang
(https://github.com/lucidrains/MEGABYTE-pytorch). The hierarchical transformer,
helper utilities and per-stage start tokens derive from that implementation.
The DNA configuration, masked-diffusion integration and Transformers model class
are added here.

> MIT License
>
> Copyright (c) 2023 Phil Wang
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### `src/phagen/generation_utils.py` and the training generation utilities

Adapts masked-diffusion generation utilities from Open-dLLM
(https://github.com/pengzhangzhi/Open-dLLM), licensed under Apache-2.0. The mixin
records further lineage from the Dream model's generation utilities.

## Training framework

PhaGen was trained with VeOmni (https://github.com/ByteDance-Seed/VeOmni), developed
by ByteDance and distributed under Apache-2.0. The adapted training framework is
retained under `training/`; original ByteDance copyright and license headers are
preserved. The project-specific DNA model and masked-diffusion training entry point
extend that framework.

The `veomni/models/loader.py` and `veomni/models/registry.py` training modules also
retain their recorded adaptation from SGLang (https://github.com/sgl-project/sglang).
Original source headers and attribution comments are preserved.

### AnyPrecisionAdamW

`training/veomni/optim/optimizer.py` retains the Meta copyright and Llama 2
Community License Agreement header on its AnyPrecisionAdamW implementation,
adapted from the Llama Recipes optimizer referenced in the source. The full
license text is included at `training/licenses/LLAMA2_LICENSE.txt`, obtained from
https://github.com/meta-llama/llama/blob/main/LICENSE. These upstream terms remain
in effect for that component; the paper's default optimizer is standard AdamW.

> Llama 2 is licensed under the LLAMA 2 Community License,
> Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## Scope

The Apache-2.0 grant applies to project-authored code alongside the upstream terms
above. It does not relicense third-party source or biological datasets. Model
weights are distributed separately on Hugging Face.
