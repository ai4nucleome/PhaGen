import os
import time
import json
import wandb
import warnings
from tqdm import trange
from functools import partial
from typing import Any, Dict, List, Optional
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist

from veomni.utils import helper
from veomni.utils.dist_utils import all_reduce
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.data import build_chat_template, build_dataloader, build_iterative_dataset, build_mapping_dataset, build_packing_dataset

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
logger = helper.create_logger(__name__)


@dataclass
class DllmDataArguments(DataArguments):
    val_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path of validation data."},
    )


@dataclass
class DllmTrainingArguments(TrainingArguments):
    eval_steps: int = field(
        default=0,
        metadata={"help": "Evaluate every N global steps. Set 0 to disable evaluation."},
    )
    eval_anchor_seed: int = field(
        default=42,
        metadata={"help": "Seed for fixed-anchor masked validation."},
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "Run forward-only sanity checks and exit without parameter updates."},
    )
    dry_run_steps: int = field(
        default=2,
        metadata={"help": "Number of training dataloader steps to run in dry-run mode."},
    )
    dry_run_run_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run validation metrics once in dry-run mode."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.eval_steps < 0:
            raise ValueError("`eval_steps` should be non-negative.")
        if self.dry_run_steps <= 0:
            raise ValueError("`dry_run_steps` should be positive.")


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DllmDataArguments" = field(default_factory=DllmDataArguments)
    train: "DllmTrainingArguments" = field(default_factory=DllmTrainingArguments)


def build_masked_eval_batch(
    micro_batch: Dict[str, Any],
    mask_token_id: int,
    bos_token_id: Optional[int],
    eos_token_id: Optional[int],
    ignore_index: int,
    rng: "torch.Generator",
) -> Dict[str, Any]:
    if "input_ids" not in micro_batch:
        raise ValueError("`input_ids` is required in validation micro-batch.")

    input_ids = micro_batch["input_ids"].clone()
    if input_ids.ndim != 2:
        raise NotImplementedError("Validation currently supports only padded 2D `input_ids`.")

    labels = micro_batch.get("labels", micro_batch["input_ids"]).clone()
    if labels.shape != input_ids.shape:
        raise ValueError("`labels` shape must match `input_ids` shape in validation.")

    batch_size, seq_len = input_ids.shape
    mask_ratio = torch.rand((batch_size, 1), generator=rng).clamp(1 / 500, 1 - 1 / 500)
    mask_ratio_full = mask_ratio.expand(-1, seq_len).clone()
    mask_indices = torch.rand(input_ids.shape, generator=rng) < mask_ratio_full

    # Keep this consistent with DataCollatorWithPaddingMasking.
    if bos_token_id is not None:
        mask_indices |= input_ids.eq(bos_token_id)
    if eos_token_id is not None:
        mask_indices |= input_ids.eq(eos_token_id)

    input_ids[mask_indices] = mask_token_id
    labels[input_ids != mask_token_id] = ignore_index

    masked_batch = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
    }
    masked_batch["input_ids"] = input_ids
    masked_batch["labels"] = labels
    masked_batch["mask_ratio"] = mask_ratio_full
    return masked_batch


@torch.no_grad()
def run_masked_validation(
    *,
    model: "torch.nn.Module",
    val_dataloader,
    mask_seed: int,
    mask_token_id: int,
    bos_token_id: Optional[int],
    eos_token_id: Optional[int],
    ignore_index: int,
    enable_mixed_precision: bool,
) -> float:
    was_training = model.training
    model.eval()

    # Avoid triggering validation dataset repack on every eval.
    # Only reset sampler epoch when available.
    sampler = getattr(val_dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(mask_seed)

    loss_sum = torch.tensor(0.0, device="cuda", dtype=torch.float64)
    token_count = torch.tensor(0.0, device="cuda", dtype=torch.float64)

    for micro_batches in iter(val_dataloader):
        for micro_batch in micro_batches:
            masked_batch = build_masked_eval_batch(
                micro_batch=micro_batch,
                mask_token_id=mask_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                ignore_index=ignore_index,
                rng=rng,
            )
            masked_tokens = (masked_batch["labels"] != ignore_index).sum()
            if masked_tokens.item() == 0:
                continue

            masked_batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in masked_batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enable_mixed_precision):
                loss: "torch.Tensor" = model(**masked_batch).loss.mean()

            loss_sum += loss.detach().double() * masked_tokens.double()
            token_count += masked_tokens.double()

    eval_stats = torch.stack((loss_sum, token_count))
    dist.all_reduce(eval_stats, op=dist.ReduceOp.SUM, group=get_parallel_state().fsdp_group)

    if was_training:
        model.train()

    if eval_stats[1].item() == 0:
        return float("nan")
    return (eval_stats[0] / eval_stats[1]).item()


@torch.no_grad()
def run_dry_run(
    *,
    model: "torch.nn.Module",
    train_dataloader,
    val_dataloader,
    model_config,
    tokenizer,
    args: "Arguments",
) -> None:
    model.eval()
    data_iterator = iter(train_dataloader)

    logger.info_rank0(f"[DryRun] start, steps={args.train.dry_run_steps}, run_eval={args.train.dry_run_run_eval}")
    train_start = time.perf_counter()
    for step_idx in range(args.train.dry_run_steps):
        step_start = time.perf_counter()
        try:
            micro_batches: List[Dict[str, Any]] = next(data_iterator)
        except StopIteration:
            logger.info_rank0(f"[DryRun] train dataloader exhausted at step {step_idx}.")
            break

        step_loss = 0.0
        for micro_batch in micro_batches:
            micro_batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.train.enable_mixed_precision):
                loss: "torch.Tensor" = model(**micro_batch).loss.mean() / len(micro_batches)
            step_loss += loss.item()

        step_loss = all_reduce(step_loss, group=get_parallel_state().fsdp_group)
        step_time = time.perf_counter() - step_start
        logger.info_rank0(
            f"[DryRun][Train {step_idx + 1}/{args.train.dry_run_steps}] "
            f"loss={step_loss:.4f}, micro_batches={len(micro_batches)}, step_time={step_time:.2f}s"
        )
    train_time = time.perf_counter() - train_start
    logger.info_rank0(f"[DryRun][Train] total_time={train_time:.2f}s")

    if val_dataloader is not None and args.train.dry_run_run_eval:
        logger.info_rank0(f"[DryRun][Eval] start, run_eval={args.train.dry_run_run_eval}")
        ignore_index = getattr(model_config, "IGNORE_INDEX", -100)
        eval_start = time.perf_counter()

        fixed_start = time.perf_counter()
        fixed_anchor_loss = run_masked_validation(
            model=model,
            val_dataloader=val_dataloader,
            mask_seed=args.train.eval_anchor_seed,
            mask_token_id=tokenizer.mask_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            ignore_index=ignore_index,
            enable_mixed_precision=args.train.enable_mixed_precision,
        )
        fixed_time = time.perf_counter() - fixed_start

        random_start = time.perf_counter()
        random_once_loss = run_masked_validation(
            model=model,
            val_dataloader=val_dataloader,
            mask_seed=args.train.seed + 1,
            mask_token_id=tokenizer.mask_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            ignore_index=ignore_index,
            enable_mixed_precision=args.train.enable_mixed_precision,
        )
        random_time = time.perf_counter() - random_start
        eval_time = time.perf_counter() - eval_start
        logger.info_rank0(
            f"[DryRun][Eval] fixed_anchor_loss={fixed_anchor_loss:.4f}, random_once_loss={random_once_loss:.4f}, "
            f"fixed_time={fixed_time:.2f}s, random_time={random_time:.2f}s, total_eval_time={eval_time:.2f}s"
        )

    logger.info_rank0("[DryRun] completed without optimizer update.")


def main():
    args = parse_args(Arguments)
    logger.info(
        f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}"
    )  # 输出当前进程的rank和world size
    logger.info_rank0(json.dumps(asdict(args), indent=2))  # 只在主进程输出args
    torch.cuda.set_device(
        f"cuda:{args.train.local_rank}"
    )  # 按照 local rank 的编号，在当前节点设置当前进程的device
    dist.init_process_group(backend="nccl")  # 初始化分布式训练环境
    # 设置随机种子
    # args.train.enable_full_determinism：是否启用全确定性，
    # 如果设置为 true，在设置随机种子时会启用 transformer 库中的 enable_full_determinism 函数(使训练过程完全可重现);
    # 设置为 false 时，会启用 transformer 库中的 set_seed 函数
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()  # 只在每个节点主进程启用第三方日志

    if args.train.global_rank == 0:
        save_args(
            args, args.train.output_dir
        )  # 只在所有节点的主进程保存args，保存到 veomni_cli.yaml 文件中

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager
    )  # 构建检查点管理器

    # 四种数据并行模式
    ## ddp：分布式数据并行；每个GPU保存完整的模型副本；梯度在反向传播之后进行 AllReduce 同步；内存使用量=模型大小 times GPU数量。适合在模型相对较小，可以完全放入单个 GPU，对内存效率要求不高的场景
    ## fsdp1：参数分片，模型的参数在多个 GPU 之间分片进行存储；梯度分片，梯度也进行分片，减少内存使用；动态收集，在前向传播时收集参数，计算后进行释放；内存使用量=模型大小 divide GPU数量；显著减小内存使用
    ## fsdp2：pytorch 原生的 fsdp2；更加高效的分片策略，支持更细粒度的分片控制；更好的性能，优化通信和计算重叠；支持 DTensor，可以使用分布式张量
    ## fsdp2-vescale：基于字节的 VeScale 框架实现的 FSDP2；支持更多的优化和自定义的功能；拓展性更好，针对大规模训练；更好的精度和性能平衡
    # 两种 checkpoint 管理方式
    ## ByteCheckpoint：支持DDP、FSDP1、FSDP2等多种并行模式；高效存储，优化的checkpoint格式和压缩
    ## DCP：PyTorch原生；使用PyTorch标准API；与PyTorch生态系统深度集成

    # 训练并行状态初始化
    init_parallel_state(
        dp_size=args.train.data_parallel_size,  # 数据并行
        tp_size=args.train.tensor_parallel_size,  # 张量并行
        ep_size=args.train.expert_parallel_size,  # 专家并行
        pp_size=args.train.pipeline_parallel_size,  # 流水线并行
        cp_size=args.train.context_parallel_size,  # 上下文并行
        ulysses_size=args.train.ulysses_parallel_size,  # 一种特殊的并行模式，用于处理长序列
        dp_mode=args.train.data_parallel_mode,  # 数据并行模式
    )

    logger.info_rank0("Prepare data")  # 主进程输出准备数据
    tokenizer = build_tokenizer(args.model.tokenizer_path)  # 构建tokenizer

    if args.data.val_path and args.train.eval_steps > 0 and not args.train.enable_masking:
        raise ValueError("Masked validation expects `train.enable_masking=true` for consistency with training.")

    # 检查模型的 tokenizer 有没有 mask 和 pad token id
    if (args.train.enable_masking or (args.data.val_path and args.train.eval_steps > 0)) and tokenizer.mask_token is None:
        tokenizer.add_special_tokens(
            {"mask_token": "[MASK]"}
        )  # 如果没有 mask token，添加 [MASK] 作为 mask token
        logger.info_rank0("Adding [MASK] as mask_token")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        logger.info_rank0("Adding [PAD] as pad_token")

    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    # 两种数据加载器类型
    ## native-iterable：可迭代数据集，把数据分批次加载到内存中
    ## native-mapping：映射数据集，把数据直接全部加载到内存中
    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(
                args.data.train_path, transform=transform, seed=args.train.seed
            )
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(
                args.data.train_path, transform=transform
            )
            args.train.compute_train_steps(
                args.data.max_seq_len, args.data.train_size, len(train_dataset)
            )
        elif args.data.datasets_type == "packing":
            logger.info_rank0("Start building packing dataset (Best Fit bin-packing)")
            train_dataset = build_packing_dataset(
                data_path=args.data.train_path,
                tokenizer=tokenizer,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
                seed=args.train.seed,
            )
            args.train.compute_train_steps(
                args.data.max_seq_len, args.data.train_size, len(train_dataset)
            )

        # 构建 dataloader
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,  # Samples per device in one forward pass; larger global batches use gradient accumulation.
            global_batch_size=args.train.global_batch_size,  # 在计算一次反向传播之前，需要积累到global_batchsize的样本个数
            dataloader_batch_size=args.train.dataloader_batch_size,  # equal to global_batch_size / data_parallel_size, also equal to micro_batchsize * grad_accumulate_steps
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,  # 是否启用无填充训练/使用位置 ID 的无填充训练，rmpad 表示无填充训练
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,  # rmpad_with_pos_ids 表示使用位置 ID 的无填充训练
            enable_masking=args.train.enable_masking,
            mask_token_id=tokenizer.mask_token_id,
            padding_token_id=tokenizer.pad_token_id,  # 获取 padding token id
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,  # 默认是 0
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,  # 默认是 200
            dyn_bsz_margin=args.train.dyn_bsz_margin,  # 默认是 0
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,  # 默认是 200
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,  # 默认是 true
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(
            f"Unsupported dataloader type: {args.data.dataloader_type}."
        )

    val_dataloader = None
    if args.data.val_path:
        if args.train.rmpad or args.train.rmpad_with_pos_ids:
            raise NotImplementedError(
                "Validation currently supports only rmpad=false and rmpad_with_pos_ids=false."
            )
        if args.data.datasets_type == "iterable":
            raise NotImplementedError("Validation for iterable dataset is not supported now.")

        if args.data.datasets_type == "mapping":
            logger.info_rank0("Start building validation mapping dataset")
            val_dataset = build_mapping_dataset(args.data.val_path, transform=transform)
        elif args.data.datasets_type == "packing":
            logger.info_rank0("Start building validation packing dataset (Best Fit bin-packing)")
            val_dataset = build_packing_dataset(
                data_path=args.data.val_path,
                tokenizer=tokenizer,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
                seed=args.train.seed + 1,
            )
        else:
            raise NotImplementedError(f"Unsupported datasets_type for validation: {args.data.datasets_type}.")

        val_dataloader = build_dataloader(
            dataset=val_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed + 1,
            max_seq_len=args.data.max_seq_len,
            train_steps=1,
            rmpad=False,
            rmpad_with_pos_ids=False,
            enable_masking=False,
            mask_token_id=tokenizer.mask_token_id,
            padding_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bsz_warmup_ratio=0.0,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=False,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
        logger.info_rank0(f"=================================================================================================")
        logger.info_rank0(f"validation samples num: {len(val_dataset)}")
    elif args.train.eval_steps > 0:
        logger.warning_rank0("`train.eval_steps` > 0 but `data.val_path` is empty. Evaluation will be skipped.")

    logger.info_rank0(f"=================================================================================================")
    logger.info_rank0(f"training samples num: {len(train_dataset)}, steps: {args.train._train_steps}")
    logger.info_rank0(f"example train_dataset[0][0]: {train_dataset[0][0].keys()}")
    logger.info_rank0(f"length of example train_dataset[0]: {len(train_dataset[0][0]['input_ids'])}")
    logger.info_rank0(f"=================================================================================================")

    # 构建模型
    logger.info_rank0("Prepare model")
    # 实例化模型，按照模型的 arch_name 来检索是否已经注册，如果arch_name在已注册的列表中，使用那个类来实例化模型对象
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    if args.train.enable_masking and len(tokenizer) != model_config.vocab_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        # 把 pad_token_id 对应的 embedding 设置为 0 向量，且在训练时不进行更新
        for embs in model.get_input_embeddings():
            embs.padding_idx = tokenizer.pad_token_id
            with torch.no_grad():
                embs.weight[tokenizer.pad_token_id].zero_()

    # 构建并行化模型
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    if args.train.dry_run:
        run_dry_run(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_config=model_config,
            tokenizer=tokenizer,
            args=args,
        )
        dist.barrier()
        dist.destroy_process_group()
        return

    # 获取优化器预钩子
    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)

    # 构建优化器
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    # 构建学习率调度器
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        # 在主进程初始化 wandb
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )
        # 在主进程初始化 profiler，来记录 GPU 和 CPU 的使用情况
        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()
        # 保存模型设置的参数
        # save model_assets before training
        model_assets = [model_config, tokenizer if args.data.data_type == "plaintext" else chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    # 初始化训练状态
    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    # 初始化训练效率监控器，用于计算和跟踪训练过程中的各种性能指标。
    environ_meter = helper.EnvironMeter(
        config=model_config,  # 模型配置
        global_batch_size=args.train.global_batch_size,  # 全局批量大小
        rmpad=args.train.rmpad,  # 是否移除填充标记
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,  # 使用position ids无填充训练
        empty_cache_steps=args.train.empty_cache_steps,  # 清空缓存的步数间隔
    )

    # 如果是断点续训，则加载检查点
    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        extra_state = state["extra_state"]
        # 获取当前训练步数
        global_step = extra_state["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        # 加载学习率调度器
        lr_scheduler.load_state_dict(extra_state["lr_scheduler"])
        # 恢复 dataloader 的精确游标，避免中断后从 epoch 头部重复读取
        if "train_dataloader" in extra_state:
            train_dataloader.load_state_dict(extra_state["train_dataloader"])
        elif start_step != 0:
            raise RuntimeError(
                "Checkpoint missing `train_dataloader` state; cannot safely resume from a mid-epoch step."
            )
        # 加载训练效率监控器
        environ_meter.load_state_dict(extra_state["environ_meter"])
        # 加载随机种子，恢复PyTorch的随机数生成器状态，确保从检查点恢复训练时能够保持随机性的一致性。
        torch.set_rng_state(extra_state["torch_rng_state"])
        # 如果训练步数为0，则清空训练数据加载器
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        # 同步进程，确保所有进程都加载了检查点
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()  # 清空缓存
    # 构建了激活值卸载上下文，用于在训练过程中管理GPU内存使用，通过将激活值（activations）卸载到CPU来节省GPU内存。
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()  # 设置模型为训练模式
    # 输出训练信息
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch") and epoch > start_epoch:
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )

        # 将数据加载器转换为迭代器，支持从任意位置开始迭代（用于从检查点恢复）
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            # 打印第一个数据示例
            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()  # 强制同步：等待所有CUDA操作完成，确保GPU和CPU之间的同步。
            start_time = time.time()
            for micro_batch in micro_batches:
                # 把当前的 micro_batch 添加到训练效率监控器中，收集训练的统计信息（token数，序列长度等）。
                # 对于文本数据，计算每条样本的长度，存储在 environ_meter.batch_seqlens 中
                environ_meter.add(micro_batch)

                # 将micro_batch中的所有张量移动到GPU，并设置non_blocking=True，允许异步操作，提高效率。
                # 只移动张量到 GPU，其他非张量数据保持不变。
                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }

                with model_fwd_context:
                    # 不使用KV缓存（训练时不需要）
                    # 平均损失，支持梯度累积
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.train.enable_mixed_precision):
                        loss: "torch.Tensor" = model(**micro_batch).loss.mean() / len(micro_batches)

                with model_bwd_context:
                    # 在激活值卸载上下文中进行；累积梯度到模型参数
                    loss.backward()

                total_loss += loss.item()
                del micro_batch  # 释放内存

            # 如果使用FSDP1模式，使用模型内置的clip_grad_norm_方法；否则使用PyTorch默认的实现。
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

            # 更新模型参数
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if hasattr(grad_norm, "full_tensor"):
                # 确保无论使用什么样的分布式训练模式，最终都能得到一个标量形式的梯度范数值
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            torch.cuda.synchronize()  # 强制同步：等待所有CUDA操作完成，确保GPU和CPU之间的同步。
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())  # 针对多模态模型做的，不同的编码器的学习率设置不同
            train_metrics = environ_meter.step(delta_time, global_step=global_step)  # 计算和收集训练过程中的各种性能指标。返回一个字典，包含各种性能指标。

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()  # 更新进度条

            if args.train.global_rank == 0:
                # 如果使用wandb，则记录训练指标
                if args.train.use_wandb:
                    train_metrics.update(
                        {
                            "training/loss": total_loss, 
                            "training/grad_norm": grad_norm, 
                            "training/lr": lr
                        }
                    )
                    wandb.log(train_metrics, step=global_step)

                # 如果启用性能分析，则在指定步数记录性能指标
                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()

            if val_dataloader is not None and args.train.eval_steps > 0 and global_step % args.train.eval_steps == 0:
                logger.info_rank0(f"[Eval step {global_step}] Run val...")
                ignore_index = getattr(model_config, "IGNORE_INDEX", -100)
                fixed_anchor_loss = run_masked_validation(
                    model=model,
                    val_dataloader=val_dataloader,
                    mask_seed=args.train.eval_anchor_seed,
                    mask_token_id=tokenizer.mask_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    ignore_index=ignore_index,
                    enable_mixed_precision=args.train.enable_mixed_precision,
                )
                random_once_loss = run_masked_validation(
                    model=model,
                    val_dataloader=val_dataloader,
                    mask_seed=args.train.seed + global_step,
                    mask_token_id=tokenizer.mask_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    ignore_index=ignore_index,
                    enable_mixed_precision=args.train.enable_mixed_precision,
                )
                logger.info_rank0(f"[Eval step {global_step}] fixed_anchor_loss={fixed_anchor_loss:.4f}, random_once_loss={random_once_loss:.4f}")
                if args.train.global_rank == 0 and args.train.use_wandb:
                    wandb.log(
                        {
                            "validation/loss_fixed_anchor": fixed_anchor_loss,
                            "validation/loss_random_once": random_once_loss,
                        },
                        step=global_step,
                    )

            # 如果启用保存步骤，则在指定步数保存检查点
            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                    model_state_dict = ckpt_to_state_dict(
                        save_checkpoint_path=save_checkpoint_path,
                        output_dir=args.train.output_dir,
                        ckpt_manager=args.train.ckpt_manager,
                    )
                    save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        # 结束一个 epoch 的训练，
        # 关闭数据加载器
        # 重置训练步数
        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        # 保存一个 epoch 的检查点
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            # save model in huggingface's format
            if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
                hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                model_state_dict = ckpt_to_state_dict(
                    save_checkpoint_path=save_checkpoint_path,
                    output_dir=args.train.output_dir,
                    ckpt_manager=args.train.ckpt_manager,
                )
                save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    # 结束所有的训练，释放内存
    # torch.cuda.synchronize() 的作用范围是单个GPU设备内部，同步GPU和CPU的操作，确保GPU操作完成之后再继续进行CPU的操作
    # dist.barrier() 是让多个进程/GPU之间同步，确保所有进程/GPU都完成一个阶段的任务后再继续进行下一个阶段的任务
    torch.cuda.synchronize()
    # 释放优化器和学习率调度器
    del optimizer, lr_scheduler
    helper.empty_cache()
    # 保存模型为huggingface的格式
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
