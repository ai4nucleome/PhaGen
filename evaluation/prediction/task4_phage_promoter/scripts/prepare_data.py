#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedGroupKFold

WINDOW_SIZE = 65
BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
RECONSTRUCTION_RULES = {
    "gc_matched_nonpromoter": "positive=centered_65bp_raw_genome_window_on_promoter_interval_oriented_with_"
    "annotated_core_resolution_for_ambiguous_genome_bases;"
    "negative=deterministic_accession_matched_real_genome_windows_avoiding_known_promoters_"
    "and_matching_positive_gc_count"
}
TASK_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PromoterAnnotation:
    promoter_name: str
    phage_name: str
    accession: str
    start: int
    end: int
    strand: str
    sequence: str
    header: str


@dataclass(frozen=True)
class GenomeRecord:
    accession: str
    record_id: str
    description: str
    sequence: str
    path: Path


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper()))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts).upper()))
    return records


def parse_promoter_header(header: str, seq: str) -> PromoterAnnotation:
    name = header.split(" promoter,", 1)[0].strip()
    phage_match = re.search(r" promoter, (.*?), (?:RefSeq|EMBL):", header)
    acc_match = re.search(r"(?:RefSeq|EMBL):([^,\s]+)", header)
    loc_match = re.search(r"(complement\(\d+\.\.\d+\)|\(\d+\.\.\d+\))", header)
    if not phage_match or not acc_match or not loc_match:
        raise ValueError(f"Could not parse promoter header: {header}")

    phage_name = phage_match.group(1)
    accession = acc_match.group(1)
    loc = loc_match.group(1)
    strand = "-" if loc.startswith("complement") else "+"
    coords = re.search(r"(\d+)\.\.(\d+)", loc)
    if coords is None:
        raise ValueError(f"Could not parse promoter coordinates: {header}")
    start = int(coords.group(1))
    end = int(coords.group(2))
    if start > end:
        raise ValueError(f"Promoter start > end in header: {header}")
    return PromoterAnnotation(
        promoter_name=name,
        phage_name=phage_name,
        accession=accession,
        start=start,
        end=end,
        strand=strand,
        sequence=seq.upper(),
        header=header,
    )


def load_promoters(path: Path) -> dict[tuple[str, str], PromoterAnnotation]:
    promoters: dict[tuple[str, str], PromoterAnnotation] = {}
    duplicate_keys: list[tuple[str, str]] = []
    for header, seq in read_fasta_records(path):
        ann = parse_promoter_header(header, seq)
        key = (ann.accession, ann.promoter_name)
        if key in promoters:
            duplicate_keys.append(key)
        promoters[key] = ann
    if duplicate_keys:
        raise ValueError(f"Duplicate promoter keys: {duplicate_keys[:10]}")
    return promoters


def load_genomes(
    accessions: Iterable[str], download_dir: Path
) -> dict[str, GenomeRecord]:
    genomes: dict[str, GenomeRecord] = {}
    for accession in accessions:
        path = download_dir / f"{accession}.fasta"
        if not path.exists():
            raise FileNotFoundError(path)
        record = SeqIO.read(path, "fasta")
        genomes[accession] = GenomeRecord(
            accession=accession,
            record_id=record.id,
            description=record.description,
            sequence=str(record.seq).upper(),
            path=path,
        )
    return genomes


def clamp_window(start0: int, genome_len: int) -> tuple[int, int, bool]:
    adjusted = False
    if genome_len < WINDOW_SIZE:
        raise ValueError(f"Genome length {genome_len} is shorter than {WINDOW_SIZE}")
    if start0 < 0:
        start0 = 0
        adjusted = True
    if start0 + WINDOW_SIZE > genome_len:
        start0 = genome_len - WINDOW_SIZE
        adjusted = True
    return start0, start0 + WINDOW_SIZE, adjusted


def positive_window(
    annotation: PromoterAnnotation, genome_seq: str
) -> tuple[int, int, str, bool]:
    promoter_start0 = annotation.start - 1
    promoter_end0 = annotation.end
    start0 = (promoter_start0 + promoter_end0 - WINDOW_SIZE) // 2
    start0, end0, adjusted = clamp_window(start0, len(genome_seq))
    seq = genome_seq[start0:end0]
    if annotation.strand == "-":
        seq = reverse_complement(seq)
    return start0 + 1, end0, seq.upper(), adjusted


def overlaps(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> bool:
    start, end = interval
    for other_start, other_end in intervals:
        if start <= other_end and other_start <= end:
            return True
    return False


def load_dataset_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def build_promoter_lookup(
    promoters: dict[tuple[str, str], PromoterAnnotation],
) -> tuple[
    dict[tuple[str, str], PromoterAnnotation], dict[tuple[str, str], PromoterAnnotation]
]:
    exact = dict(promoters)
    normalized: dict[tuple[str, str], PromoterAnnotation] = {}
    for (accession, name), ann in promoters.items():
        normalized[(accession, name.lower())] = ann
    return exact, normalized


@dataclass(frozen=True)
class CandidateWindow:
    start: int
    end: int
    sequence: str
    strand: str
    gc_count: int
    motif_score: float | None = None
    motif_rank: int | None = None


@dataclass(frozen=True)
class NegativeAssignment:
    start: int
    end: int
    sequence: str
    strand: str
    sample_seed: int
    gc_count: int
    target_gc_count: int | None
    gc_abs_delta: int | None
    motif_score: float | None
    motif_rank: int | None


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def gc_count(sequence: str) -> int:
    return sequence.count("G") + sequence.count("C")


def gc_fraction(sequence: str) -> float:
    return gc_count(sequence) / len(sequence)


def resolve_ambiguous_promoter_core(
    sequence: str,
    annotated_core: str,
) -> tuple[str, bool]:
    sequence = sequence.upper()
    annotated_core = annotated_core.upper()
    if annotated_core in sequence:
        return sequence, False

    candidates: list[tuple[int, str]] = []
    for start in range(0, len(sequence) - len(annotated_core) + 1):
        observed = sequence[start : start + len(annotated_core)]
        mismatches = [
            (observed_base, annotated_base)
            for observed_base, annotated_base in zip(observed, annotated_core)
            if observed_base != annotated_base
        ]
        if not mismatches:
            continue
        if all(
            observed_base not in BASE_TO_INDEX and annotated_base in BASE_TO_INDEX
            for observed_base, annotated_base in mismatches
        ):
            resolved = (
                sequence[:start]
                + annotated_core
                + sequence[start + len(annotated_core) :]
            )
            candidates.append((start, resolved))

    if len(candidates) > 1:
        starts = [start for start, _ in candidates]
        raise ValueError(
            f"Ambiguous promoter core has multiple resolvable placements: {starts}"
        )
    if not candidates:
        return sequence, False
    return candidates[0][1], True


def positive_sample(
    row: dict[str, str],
    dataset_name: str,
    csv_path: Path,
    negative_mode: str,
    promoters: dict[tuple[str, str], PromoterAnnotation],
    normalized_promoters: dict[tuple[str, str], PromoterAnnotation],
    genomes: dict[str, GenomeRecord],
    known_intervals: dict[str, list[tuple[int, int]]],
) -> dict[str, object]:
    sample_id = row[""]
    accession, suffix = sample_id.split(":", 1)
    annotation = promoters.get((accession, suffix)) or normalized_promoters.get(
        (accession, suffix.lower())
    )
    if annotation is None:
        raise KeyError(f"Missing promoter annotation for {sample_id}")

    genome = genomes[accession]
    window_start, window_end, sequence, coordinate_adjusted = positive_window(
        annotation,
        genome.sequence,
    )
    sequence, promoter_core_overlaid = resolve_ambiguous_promoter_core(
        sequence,
        annotation.sequence,
    )
    contains_core = annotation.sequence in sequence
    valid_acgt = bool(re.fullmatch(r"[ACGT]+", sequence))
    seq_gc_count = gc_count(sequence)
    return {
        "dataset": dataset_name,
        "negative_mode": negative_mode,
        "source_csv": csv_path.name,
        "sample_id": sample_id,
        "accession": accession,
        "label": 1,
        "y": 1,
        "window_start": window_start,
        "window_end": window_end,
        "strand": annotation.strand,
        "promoter_name": annotation.promoter_name,
        "promoter_start": annotation.start,
        "promoter_end": annotation.end,
        "promoter_length": annotation.end - annotation.start + 1,
        "promoter_strand": annotation.strand,
        "sequence": sequence,
        "seq": sequence,
        "sequence_length": len(sequence),
        "gc_count": seq_gc_count,
        "gc_fraction": seq_gc_count / len(sequence),
        "valid_acgt": int(valid_acgt),
        "contains_promoter_core": contains_core,
        "promoter_core_overlaid": int(promoter_core_overlaid),
        "overlaps_known_promoter": int(
            overlaps((window_start, window_end), known_intervals.get(accession, []))
        ),
        "coordinate_adjusted": int(coordinate_adjusted),
        "source_negative_id": "",
        "source_numeric_suffix": "",
        "negative_sampling_seed": "",
        "negative_target_gc_count": "",
        "negative_gc_abs_delta": "",
        "negative_motif_score": "",
        "negative_motif_rank": "",
        "fold": -1,
        "reconstruction_rule": RECONSTRUCTION_RULES[negative_mode],
    }


def valid_forward_candidate(
    genome_sequence: str,
    start: int,
    known_intervals: list[tuple[int, int]],
) -> CandidateWindow | None:
    end = start + WINDOW_SIZE - 1
    if overlaps((start, end), known_intervals):
        return None
    sequence = genome_sequence[start - 1 : end].upper()
    if not re.fullmatch(r"[ACGT]+", sequence):
        return None
    return CandidateWindow(
        start=start,
        end=end,
        sequence=sequence,
        strand="+",
        gc_count=gc_count(sequence),
    )


def build_gc_candidate_bins(
    accession: str,
    dataset_sampling_key: str,
    genome: GenomeRecord,
    known_intervals: dict[str, list[tuple[int, int]]],
    seed: int,
) -> dict[int, list[CandidateWindow]]:
    bins: dict[int, list[CandidateWindow]] = {}
    intervals = known_intervals.get(accession, [])
    for start in range(1, len(genome.sequence) - WINDOW_SIZE + 2):
        candidate = valid_forward_candidate(genome.sequence, start, intervals)
        if candidate is None:
            continue
        bins.setdefault(candidate.gc_count, []).append(candidate)

    rng = random.Random(stable_seed(seed, dataset_sampling_key, accession, "gc_bins"))
    for candidates in bins.values():
        rng.shuffle(candidates)
    return bins


def positive_gc_targets(
    positive_samples: list[dict[str, object]],
    accession: str,
    count: int,
    seed: int,
    dataset_sampling_key: str,
) -> list[int]:
    accession_targets = [
        int(row["gc_count"])
        for row in positive_samples
        if str(row["accession"]) == accession
    ]
    global_targets = [int(row["gc_count"]) for row in positive_samples]
    targets = accession_targets or global_targets
    if not targets:
        raise RuntimeError("Cannot build GC-matched negatives without positive targets")
    targets = list(targets)
    random.Random(
        stable_seed(seed, dataset_sampling_key, accession, "target_gc")
    ).shuffle(targets)
    return [targets[idx % len(targets)] for idx in range(count)]


def next_gc_candidate(
    bins: dict[int, list[CandidateWindow]],
    target_gc_count: int,
    used_sequences: set[str],
) -> CandidateWindow | None:
    for delta in range(WINDOW_SIZE + 1):
        gc_bins = (
            [target_gc_count]
            if delta == 0
            else [target_gc_count - delta, target_gc_count + delta]
        )
        for gc_bin in gc_bins:
            if gc_bin < 0 or gc_bin > WINDOW_SIZE:
                continue
            candidates = bins.get(gc_bin)
            if not candidates:
                continue
            while candidates:
                candidate = candidates.pop()
                if candidate.sequence in used_sequences:
                    continue
                return candidate
    return None


def assignment_from_candidate(
    candidate: CandidateWindow,
    sample_seed: int,
    target_gc_count: int | None = None,
) -> NegativeAssignment:
    gc_delta = (
        None if target_gc_count is None else abs(candidate.gc_count - target_gc_count)
    )
    return NegativeAssignment(
        start=candidate.start,
        end=candidate.end,
        sequence=candidate.sequence,
        strand=candidate.strand,
        sample_seed=sample_seed,
        gc_count=candidate.gc_count,
        target_gc_count=target_gc_count,
        gc_abs_delta=gc_delta,
        motif_score=candidate.motif_score,
        motif_rank=candidate.motif_rank,
    )


def sample_gc_matched_negative_windows(
    dataset_sampling_key: str,
    rows_by_accession: dict[str, list[dict[str, str]]],
    positive_samples: list[dict[str, object]],
    genomes: dict[str, GenomeRecord],
    known_intervals: dict[str, list[tuple[int, int]]],
    seed: int,
) -> dict[str, NegativeAssignment]:
    assignments: dict[str, NegativeAssignment] = {}
    used_sequences: set[str] = set()
    for accession, rows in sorted(rows_by_accession.items()):
        bins = build_gc_candidate_bins(
            accession,
            dataset_sampling_key,
            genomes[accession],
            known_intervals,
            seed,
        )
        targets = positive_gc_targets(
            positive_samples, accession, len(rows), seed, dataset_sampling_key
        )
        for row, target_gc_count in zip(rows, targets):
            candidate = next_gc_candidate(bins, target_gc_count, used_sequences)
            if candidate is None:
                raise RuntimeError(
                    f"Could not sample GC-matched negative for {row['']}"
                )
            used_sequences.add(candidate.sequence)
            assignments[row[""]] = assignment_from_candidate(
                candidate,
                sample_seed=stable_seed(seed, dataset_sampling_key, row[""]),
                target_gc_count=target_gc_count,
            )
    return assignments


def negative_sample(
    row: dict[str, str],
    dataset_name: str,
    csv_path: Path,
    negative_mode: str,
    assignment: NegativeAssignment,
    known_intervals: dict[str, list[tuple[int, int]]],
) -> dict[str, object]:
    sample_id = row[""]
    accession, suffix = sample_id.split(":", 1)
    return {
        "dataset": dataset_name,
        "negative_mode": negative_mode,
        "source_csv": csv_path.name,
        "sample_id": sample_id,
        "accession": accession,
        "label": 0,
        "y": 0,
        "window_start": assignment.start,
        "window_end": assignment.end,
        "strand": assignment.strand,
        "promoter_name": "",
        "promoter_start": "",
        "promoter_end": "",
        "promoter_length": "",
        "promoter_strand": "",
        "sequence": assignment.sequence,
        "seq": assignment.sequence,
        "sequence_length": len(assignment.sequence),
        "gc_count": assignment.gc_count,
        "gc_fraction": assignment.gc_count / len(assignment.sequence),
        "valid_acgt": 1,
        "contains_promoter_core": "",
        "promoter_core_overlaid": "",
        "overlaps_known_promoter": int(
            overlaps(
                (assignment.start, assignment.end), known_intervals.get(accession, [])
            )
        ),
        "coordinate_adjusted": 0,
        "source_negative_id": sample_id,
        "source_numeric_suffix": suffix if suffix.isdigit() else "",
        "negative_sampling_seed": assignment.sample_seed,
        "negative_target_gc_count": ""
        if assignment.target_gc_count is None
        else assignment.target_gc_count,
        "negative_gc_abs_delta": ""
        if assignment.gc_abs_delta is None
        else assignment.gc_abs_delta,
        "negative_motif_score": ""
        if assignment.motif_score is None
        else assignment.motif_score,
        "negative_motif_rank": ""
        if assignment.motif_rank is None
        else assignment.motif_rank,
        "fold": -1,
        "reconstruction_rule": RECONSTRUCTION_RULES[negative_mode],
    }


def assign_group_folds(rows: list[dict[str, object]], n_splits: int, seed: int) -> None:
    y = [int(row["label"]) for row in rows]
    groups = [str(row["accession"]) for row in rows]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(rows, y, groups)):
        for idx in test_idx:
            rows[int(idx)]["fold"] = int(fold)
    missing = [row["sample_id"] for row in rows if int(row["fold"]) < 0]
    if missing:
        raise RuntimeError(f"Failed to assign folds for {len(missing)} rows")


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def build_accession_components(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, int]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--similarity-threshold must be in (0, 1]")

    accessions = sorted(df["accession"].unique().tolist())
    union_find = UnionFind(accessions)
    positives = df.loc[df["y"].eq(1), ["sample_id", "accession", "seq"]].reset_index(
        drop=True
    )
    sequence_length = int(positives["seq"].str.len().iloc[0])
    minimum_matches = int(math.ceil(threshold * sequence_length - 1e-12))
    sequences = positives["seq"].tolist()
    reverse_sequences = [reverse_complement(sequence) for sequence in sequences]
    edges: list[dict[str, object]] = []

    for left_idx in range(len(positives)):
        left_sequence = sequences[left_idx]
        left_accession = str(positives.at[left_idx, "accession"])
        for right_idx in range(left_idx + 1, len(positives)):
            right_accession = str(positives.at[right_idx, "accession"])
            if left_accession == right_accession:
                continue
            direct_matches = sum(
                a == b for a, b in zip(left_sequence, sequences[right_idx])
            )
            reverse_matches = sum(
                a == b for a, b in zip(left_sequence, reverse_sequences[right_idx])
            )
            best_matches = max(direct_matches, reverse_matches)
            if best_matches < minimum_matches:
                continue
            orientation = (
                "forward" if direct_matches >= reverse_matches else "reverse_complement"
            )
            identity = best_matches / sequence_length
            union_find.union(left_accession, right_accession)
            edges.append(
                {
                    "left_sample_id": positives.at[left_idx, "sample_id"],
                    "left_accession": left_accession,
                    "right_sample_id": positives.at[right_idx, "sample_id"],
                    "right_accession": right_accession,
                    "best_orientation": orientation,
                    "matching_bases": best_matches,
                    "mismatching_bases": sequence_length - best_matches,
                    "sequence_length": sequence_length,
                    "identity": identity,
                }
            )

    members_by_root: dict[str, list[str]] = {}
    for accession in accessions:
        members_by_root.setdefault(union_find.find(accession), []).append(accession)
    ordered_components = sorted(
        tuple(sorted(members)) for members in members_by_root.values()
    )
    accession_to_cluster: dict[str, str] = {}
    membership_rows: list[dict[str, object]] = []
    for component_idx, members in enumerate(ordered_components, start=1):
        cluster_id = f"APS95_{component_idx:03d}"
        component_df = df[df["accession"].isin(members)]
        for accession in members:
            accession_to_cluster[accession] = cluster_id
            accession_df = df[df["accession"].eq(accession)]
            membership_rows.append(
                {
                    "cluster_id": cluster_id,
                    "accession": accession,
                    "cluster_accession_count": len(members),
                    "cluster_row_count": int(len(component_df)),
                    "cluster_positive_count": int(component_df["y"].sum()),
                    "accession_row_count": int(len(accession_df)),
                    "accession_positive_count": int(accession_df["y"].sum()),
                }
            )

    edge_df = pd.DataFrame(edges)
    membership_df = pd.DataFrame(membership_rows)
    return accession_to_cluster, edge_df, membership_df, minimum_matches


def rebuild(raw_dir):
    csv_path = raw_dir / "DATASET2400.csv"
    raw_rows = load_dataset_rows(csv_path)
    promoters = load_promoters(raw_dir / "promoters.fasta")
    accessions = sorted({row[""].split(":", 1)[0] for row in raw_rows})
    genomes = load_genomes(accessions, raw_dir / "genomes")
    known_intervals = {}
    for annotation in promoters.values():
        known_intervals.setdefault(annotation.accession, []).append(
            (annotation.start, annotation.end)
        )
    exact, normalized = build_promoter_lookup(promoters)
    mode = "gc_matched_nonpromoter"
    dataset = "dataset2400_gc_matched_nonpromoter"
    positives = {
        row[""]: positive_sample(
            row, dataset, csv_path, mode, exact, normalized, genomes, known_intervals
        )
        for row in raw_rows
        if int(row["output"]) == 1
    }
    negative_rows = {}
    for row in raw_rows:
        if int(row["output"]) == 0:
            negative_rows.setdefault(row[""].split(":", 1)[0], []).append(row)
    assignments = sample_gc_matched_negative_windows(
        dataset,
        negative_rows,
        list(positives.values()),
        genomes,
        known_intervals,
        seed=42,
    )
    rows = [
        positives[row[""]]
        if int(row["output"]) == 1
        else negative_sample(
            row, dataset, csv_path, mode, assignments[row[""]], known_intervals
        )
        for row in raw_rows
    ]
    rows = [
        row
        for row in rows
        if int(row["valid_acgt"]) == 1
        and (int(row["label"]) == 0 or row["contains_promoter_core"] is True)
    ]
    assign_group_folds(rows, n_splits=5, seed=42)
    import io

    source = io.StringIO()
    writer = csv.DictWriter(
        source, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    source.seek(0)
    df = pd.read_csv(source, sep="\t")
    mapping, edges, membership, minimum_matches = build_accession_components(
        df, threshold=0.95
    )
    df["source_fold"] = df["fold"].astype(int)
    df["cluster_id"] = df["accession"].map(mapping)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=11)
    for fold, (_, test_idx) in enumerate(splitter.split(df, df["y"], df["cluster_id"])):
        df.loc[test_idx, "fold"] = fold
    df["split_protocol"] = "accession_components_joined_by_positive_promoter_similarity"
    df["promoter_similarity_threshold"] = 0.95
    df["promoter_similarity_metric"] = (
        "ungapped_identity_max_forward_reverse_complement"
    )
    df["fold_seed_selection"] = "best_label_count_balance_seed_0_to_999"
    df["fold_seed"] = 11
    qc = {
        "n_samples": len(df),
        "window_length": 65,
        "n_accessions": df["accession"].nunique(),
        "n_clusters": df["cluster_id"].nunique(),
        "negative_sampling_seed": 42,
        "fold_seed": 11,
        "minimum_promoter_matches": minimum_matches,
        "similar_promoter_pairs": len(edges),
        "positive_count": int(df["y"].sum()),
        "negative_count": int((df["y"] == 0).sum()),
        "promoter_core_overlaid": int(df["promoter_core_overlaid"].fillna(0).sum()),
        "folds": [
            {
                "fold": int(f),
                "n": len(g),
                "positive": int(g["y"].sum()),
                "negative": int((g["y"] == 0).sum()),
            }
            for f, g in df.groupby("fold")
        ],
    }
    return df, edges, membership, qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=TASK_DIR / "data" / "raw")
    parser.add_argument(
        "--output_dir", type=Path, default=TASK_DIR / "data" / "processed"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = [
        "phage_promoter_dataset2400_65bp.tsv",
        "promoter_similarity_edges.tsv",
        "cluster_membership.tsv",
        "dataset_qc.json",
    ]
    if not args.force and any((args.output_dir / name).exists() for name in names):
        raise FileExistsError("Processed benchmark exists; use --force to replace it.")
    df, edges, membership, qc = rebuild(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame, name in zip((df, edges, membership), names):
        frame.to_csv(args.output_dir / name, sep="\t", index=False)
    (args.output_dir / names[3]).write_text(json.dumps(qc, indent=2) + "\n")
    print(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
