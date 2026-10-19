#!/usr/bin/env python3
"""Split a parquet dataset into train/val parquet files.

Example:
  python training/scripts/split_parquet.py \
    --input /path/to/megadna_training_dataset.parquet \
    --val-ratio 0.05 \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _ratio_to_suffix(ratio: float) -> str:
    return f"{int(round(ratio * 100)):02d}"


def _default_output_paths(input_path: Path, val_ratio: float, seed: int) -> tuple[Path, Path]:
    stem = input_path.stem
    suffix = input_path.suffix or ".parquet"
    train_ratio = 1.0 - val_ratio
    train_path = input_path.with_name(
        f"{stem}_train{_ratio_to_suffix(train_ratio)}_seed{seed}{suffix}"
    )
    val_path = input_path.with_name(
        f"{stem}_val{_ratio_to_suffix(val_ratio)}_seed{seed}{suffix}"
    )
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a parquet file into train/val parquet files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet path (file or directory readable by pyarrow.dataset).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio in (0, 1), e.g. 0.05.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--train-output",
        default=None,
        help="Output parquet path for train split. If omitted, auto-generate next to input.",
    )
    parser.add_argument(
        "--val-output",
        default=None,
        help="Output parquet path for val split. If omitted, auto-generate next to input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for streaming split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def ensure_writable(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")


def _paths_alias(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def _validate_output_paths(
    input_path: Path,
    train_output: Path,
    val_output: Path,
    input_files: list[str],
) -> None:
    if _paths_alias(train_output, val_output):
        raise ValueError("--train-output and --val-output must be different files.")
    for output in (train_output, val_output):
        if input_path.is_dir() and (
            output.absolute().is_relative_to(input_path.absolute())
            or output.resolve().is_relative_to(input_path.resolve())
        ):
            raise ValueError(f"Output must be outside the input dataset directory: {output}")
        if any(_paths_alias(output, Path(source)) for source in input_files):
            raise ValueError(f"Output must not overwrite an input dataset file: {output}")


def split_parquet(
    input_path: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    dataset = ds.dataset(str(input_path), format="parquet")
    _validate_output_paths(input_path, train_output, val_output, dataset.files)
    scanner = dataset.scanner(batch_size=batch_size)

    rng = np.random.default_rng(seed)
    train_writer = None
    val_writer = None
    train_rows = 0
    val_rows = 0

    try:
        for record_batch in scanner.to_batches():
            table = pa.Table.from_batches([record_batch])
            num_rows = table.num_rows
            if num_rows == 0:
                continue

            pick_val = rng.random(num_rows) < val_ratio
            val_indices = np.nonzero(pick_val)[0]
            train_indices = np.nonzero(~pick_val)[0]

            if train_indices.size > 0:
                train_table = table.take(pa.array(train_indices, type=pa.int64()))
                if train_writer is None:
                    train_writer = pq.ParquetWriter(
                        str(train_output), train_table.schema, compression="snappy"
                    )
                train_writer.write_table(train_table)
                train_rows += train_table.num_rows

            if val_indices.size > 0:
                val_table = table.take(pa.array(val_indices, type=pa.int64()))
                if val_writer is None:
                    val_writer = pq.ParquetWriter(
                        str(val_output), val_table.schema, compression="snappy"
                    )
                val_writer.write_table(val_table)
                val_rows += val_table.num_rows
    finally:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    total = train_rows + val_rows
    if total == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")

    return {
        "train_rows": float(train_rows),
        "val_rows": float(val_rows),
        "total_rows": float(total),
        "val_ratio_actual": float(val_rows / total),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError(f"--val-ratio must be in (0, 1), got {args.val_ratio}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.train_output is None or args.val_output is None:
        default_train, default_val = _default_output_paths(input_path, args.val_ratio, args.seed)
    else:
        default_train, default_val = None, None

    train_output = Path(args.train_output) if args.train_output else default_train
    val_output = Path(args.val_output) if args.val_output else default_val

    ensure_writable(train_output, overwrite=args.overwrite)
    ensure_writable(val_output, overwrite=args.overwrite)

    metrics = split_parquet(
        input_path=input_path,
        train_output=train_output,
        val_output=val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print(f"input_path: {input_path}")
    print(f"train_output: {train_output}")
    print(f"val_output: {val_output}")
    print(f"seed: {args.seed}")
    print(f"target_val_ratio: {args.val_ratio}")
    print(f"train_rows: {int(metrics['train_rows'])}")
    print(f"val_rows: {int(metrics['val_rows'])}")
    print(f"total_rows: {int(metrics['total_rows'])}")
    print(f"actual_val_ratio: {metrics['val_ratio_actual']:.6f}")


if __name__ == "__main__":
    main()
