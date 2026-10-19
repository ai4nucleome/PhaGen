

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from BCBio import GFF
from Bio import SeqIO

TASK1_ROOT = Path(__file__).resolve().parent.parent
PHAGE_CODE = {"lambda": "NC_001416", "p1": "NC_005856"}
STOP_CODON_CASSETTE = "TAATAATAATAGTGA"
STOP_CODON_CASSETTE_RC = "TCACTATTATTATTA"
INSERT_OFFSET = 12
CASSETTE_LEN = 15
CONTEXT_BP = 8192
_COMPLEMENT = str.maketrans("ATCGatcg", "TAGCtagc")


def default_data_dir() -> Path:
    return TASK1_ROOT / "data"


def default_results_dir() -> Path:
    return TASK1_ROOT / "results"


def default_essen_csv(phage_name: str) -> Path:
    return (
        default_data_dir()
        / phage_name
        / f"{phage_name}_gene_essentiality_CORRECTED.csv"
    )


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    records = list(SeqIO.parse(str(path), "fasta"))
    return [record.id for record in records], [
        str(record.seq).upper() for record in records
    ]


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def load_gene_annotations(
    gff_path: str, essen_csv: str, gff_type: str = "CDS"
) -> list[dict]:
    df = pd.read_csv(essen_csv, sep="\t")
    df["Essentiality"] = df["Essentiality"].astype(str).str.strip().str.upper()
    essential = set(df[df["Essentiality"] == "E"]["locus_tag"].str.strip())
    nonessential = set(df[df["Essentiality"] == "NE"]["locus_tag"].str.strip())
    genes: list[dict] = []
    with open(gff_path) as fh:
        for rec in GFF.parse(fh, limit_info=dict(gff_type=[gff_type])):
            for feat in rec.features:
                ltag = feat.qualifiers.get("locus_tag", [""])[0].strip()
                if not ltag:
                    continue
                gene_name = feat.qualifiers.get("gene", [""])[0].strip()
                start = int(feat.location.start)
                end = int(feat.location.end)
                strand = 1 if feat.location.strand in (1, "+") else -1
                if ltag in essential:
                    label = 1
                elif ltag in nonessential:
                    label = 0
                else:
                    label = None
                genes.append(
                    {
                        "locus_tag": ltag,
                        "gene_name": gene_name,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "label": label,
                    }
                )
    gff_locus_tags = {g["locus_tag"] for g in genes}
    all_csv_tags = essential | nonessential
    missing = sorted(all_csv_tags - gff_locus_tags)
    if missing:
        print(
            f"  [WARN] {len(missing)} locus_tag(s) in essentiality CSV but absent from GFF3 ({gff_type}): {missing}"
        )
    return genes


def assign_labels(genes: list[dict], essen_csv: str) -> list[dict]:
    df = pd.read_csv(essen_csv, sep="\t")
    df["Essentiality"] = df["Essentiality"].astype(str).str.strip().str.upper()
    essential = set(df[df["Essentiality"] == "E"]["locus_tag"].str.strip())
    nonessential = set(df[df["Essentiality"] == "NE"]["locus_tag"].str.strip())
    for g in genes:
        ltag = g["locus_tag"]
        if ltag in essential:
            g["label"] = 1
        elif ltag in nonessential:
            g["label"] = 0
        else:
            g["label"] = None
    return genes


def validate_paper_protocol(
    context_bp: int = CONTEXT_BP,
    mutation_mode: str = "insert",
    mutation_offset_nt: int = INSERT_OFFSET,
    cassette_nt: int = CASSETTE_LEN,
) -> None:
    if (context_bp, mutation_mode, mutation_offset_nt, cassette_nt) != (
        8192,
        "insert",
        12,
        15,
    ):
        raise ValueError(
            "The paper protocol uses insert/8192 bp/offset 12 nt/cassette 15 nt."
        )


def _build_mutated_gene_fwd(
    gene_fwd: str,
    strand: int,
    mutation_mode: str = "insert",
    mutation_offset_nt: int = INSERT_OFFSET,
    cassette_nt: int = CASSETTE_LEN,
) -> tuple[str, int, int]:
    validate_paper_protocol(
        mutation_mode=mutation_mode,
        mutation_offset_nt=mutation_offset_nt,
        cassette_nt=cassette_nt,
    )
    if strand == 1:
        position = min(mutation_offset_nt, len(gene_fwd))
        cassette = STOP_CODON_CASSETTE
    else:
        position = max(0, len(gene_fwd) - mutation_offset_nt)
        cassette = STOP_CODON_CASSETTE_RC
    return gene_fwd[:position] + cassette + gene_fwd[position:], position, len(cassette)


def _assemble_with_context(
    genome_seq: str,
    gene_start: int,
    gene_end: int,
    core: str,
    strand: int,
    context_bp: int,
) -> tuple[str, int, int, int, int]:
    core_len = len(core)
    genome_len = len(genome_seq)
    if core_len >= context_bp:
        if strand == -1:
            seq = core[-context_bp:]
        else:
            seq = core[:context_bp]
        return (seq, 0, context_bp, 0, 0)
    total_flank = context_bp - core_len
    left_want = total_flank // 2
    right_want = total_flank - left_want
    left_avail = gene_start
    right_avail = genome_len - gene_end
    left_ctx = min(left_want, left_avail)
    right_ctx = min(right_want, right_avail)
    if left_ctx < left_want:
        right_ctx = min(right_ctx + (left_want - left_ctx), right_avail)
    elif right_ctx < right_want:
        left_ctx = min(left_ctx + (right_want - right_ctx), left_avail)
    seq = (
        genome_seq[gene_start - left_ctx : gene_start]
        + core
        + genome_seq[gene_end : gene_end + right_ctx]
    )
    return (seq, left_ctx, core_len, left_ctx, right_ctx)


def build_wt_and_mut_inputs(
    genome_seq: str,
    gene_start: int,
    gene_end: int,
    strand: int,
    context_bp: int | None,
    mutation_mode: str = "insert",
    mutation_offset_nt: int = INSERT_OFFSET,
    cassette_nt: int = CASSETTE_LEN,
) -> dict:
    validate_paper_protocol(context_bp, mutation_mode, mutation_offset_nt, cassette_nt)
    gene_fwd = genome_seq[gene_start:gene_end]
    (mut_gene_fwd, ins_in_gene, cassette_span_nt) = _build_mutated_gene_fwd(
        gene_fwd,
        strand,
        mutation_mode=mutation_mode,
        mutation_offset_nt=mutation_offset_nt,
        cassette_nt=cassette_nt,
    )
    (wt_seq, wt_off, wt_len, wt_l, wt_r) = _assemble_with_context(
        genome_seq, gene_start, gene_end, gene_fwd, strand, context_bp
    )
    (mut_seq, mut_off, mut_len, mut_l, mut_r) = _assemble_with_context(
        genome_seq, gene_start, gene_end, mut_gene_fwd, strand, context_bp
    )
    if len(mut_gene_fwd) >= context_bp:
        if strand == -1:
            shift = len(mut_gene_fwd) - context_bp
            insert_pos_in_mut_seq = max(0, min(ins_in_gene - shift, context_bp))
        else:
            insert_pos_in_mut_seq = max(0, min(ins_in_gene, context_bp))
    else:
        insert_pos_in_mut_seq = mut_off + ins_in_gene
    return {
        "wt_seq": wt_seq,
        "wt_gene_offset": wt_off,
        "wt_gene_len": wt_len,
        "wt_left_ctx": wt_l,
        "wt_right_ctx": wt_r,
        "mut_seq": mut_seq,
        "mut_gene_offset": mut_off,
        "mut_gene_len": mut_len,
        "mut_left_ctx": mut_l,
        "mut_right_ctx": mut_r,
        "insert_pos_in_mut_seq": insert_pos_in_mut_seq,
        "mutation_mode": mutation_mode,
        "mutation_offset_nt": int(mutation_offset_nt),
        "cassette_len_nt": int(cassette_nt),
        "cassette_len_nt_actual": int(cassette_span_nt),
    }


def get_coding_interval_positions(
    gene_offset: int, gene_len: int, strand: int, coding_start: int, interval_len: int
) -> list[int]:
    start = max(0, int(coding_start))
    end = min(start + max(0, int(interval_len)), int(gene_len))
    if end <= start:
        return []
    if strand == 1:
        return list(range(gene_offset + start, gene_offset + end))
    return [gene_offset + (gene_len - 1 - c) for c in range(start, end)]


def get_insert_cassette_pairs(inp: dict, strand: int) -> tuple[list[int], list[int]]:
    wt_gene_offset = int(inp["wt_gene_offset"])
    wt_gene_len = int(inp["wt_gene_len"])
    mut_gene_offset = int(inp["mut_gene_offset"])
    mut_gene_len = int(inp["mut_gene_len"])
    cassette_span_nt = int(inp.get("cassette_len_nt_actual", CASSETTE_LEN))
    rel_insert_forward = int(inp["insert_pos_in_mut_seq"]) - mut_gene_offset
    if strand == 1:
        insert_coding_offset = rel_insert_forward
    else:
        insert_coding_offset = wt_gene_len - rel_insert_forward
    insert_coding_offset = max(0, min(insert_coding_offset, wt_gene_len))
    n_cass = max(
        0,
        min(
            cassette_span_nt,
            wt_gene_len - insert_coding_offset,
            mut_gene_len - insert_coding_offset,
        ),
    )
    wt_positions = get_coding_interval_positions(
        wt_gene_offset, wt_gene_len, strand, insert_coding_offset, n_cass
    )
    mut_positions = get_coding_interval_positions(
        mut_gene_offset, mut_gene_len, strand, insert_coding_offset, n_cass
    )
    return (wt_positions, mut_positions)


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--phage", choices=["lambda", "p1"], default="lambda")
    parser.add_argument("--context", nargs="+", choices=["8192"], default=["8192"])
    parser.add_argument("--mutation-mode", choices=["insert"], default="insert")
    parser.add_argument("--mutation-offset-nt", type=int, choices=[12], default=12)
    parser.add_argument("--cassette-nt", type=int, choices=[15], default=15)
    parser.add_argument("--gff-type", choices=["CDS"], default="CDS")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-genes",
        type=int,
        default=0,
        help="Small-run limit; 0 evaluates all labelled CDSs.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing result file."
    )
    return parser


def load_inputs(args) -> tuple[str, list[dict]]:
    directory = Path(args.data_dir) / args.phage
    code = PHAGE_CODE[args.phage]
    _, sequences = read_fasta(directory / f"{code}.1.fasta")
    if len(sequences) != 1:
        raise ValueError("The phage FASTA must contain exactly one genome.")
    genes = load_gene_annotations(
        str(directory / f"{code}.1.gff3"),
        str(directory / f"{args.phage}_gene_essentiality_CORRECTED.csv"),
        gff_type="CDS",
    )
    genes = [gene for gene in genes if gene["label"] is not None]
    expected = {"lambda": (67, 30), "p1": (107, 27)}[args.phage]
    if (len(genes), sum(gene["label"] for gene in genes)) != expected:
        raise ValueError(f"Unexpected labelled CDS cohort for {args.phage}.")
    if args.max_genes < 0:
        raise ValueError("--max-genes must be non-negative.")
    if args.max_genes:
        genes = genes[: args.max_genes]
    return sequences[0], genes


def gene_result(
    gene: dict, inputs: dict, wt_score: float, mut_score: float, **extra
) -> dict:
    return {
        "locus_tag": gene["locus_tag"],
        "gene": gene["gene_name"],
        "start": gene["start"],
        "end": gene["end"],
        "strand": gene["strand"],
        "label": gene["label"],
        "wt_score": wt_score,
        "mut_score": mut_score,
        "delta": wt_score - mut_score,
        "wt_len_nt": len(inputs["wt_seq"]),
        "mut_len_nt": len(inputs["mut_seq"]),
        "wt_gene_offset": inputs["wt_gene_offset"],
        "mut_gene_offset": inputs["mut_gene_offset"],
        "insert_pos_in_mut_seq": inputs["insert_pos_in_mut_seq"],
        "mutation_offset_nt": 12,
        "cassette_len_nt": 15,
        "cassette_len_nt_actual": inputs["cassette_len_nt_actual"],
        "wt_left_context_bp": inputs["wt_left_ctx"],
        "wt_right_context_bp": inputs["wt_right_ctx"],
        "mut_left_context_bp": inputs["mut_left_ctx"],
        "mut_right_context_bp": inputs["mut_right_ctx"],
        **extra,
    }


def result_payload(
    args, model_name: str, genes: list[dict], score_type: str, **extra
) -> dict:
    return {
        "method": "evo_style_stop_codon",
        "mutation_mode": "multistop12_insert",
        "mutation_offset_nt": 12,
        "cassette_len_nt": 15,
        "context_bp": 8192,
        "model_name": model_name,
        "phage_name": args.phage,
        "gff_type": "CDS",
        "score_type": score_type,
        "genes": genes,
        **extra,
    }


def build_output_path(
    results_dir: Path,
    phage_name: str,
    model_name: str,
    context_bp: int = CONTEXT_BP,
    step_id: str | int | None = None,
    gff_type: str = "CDS",
    mutation_mode: str = "insert",
    mutation_offset_nt: int = INSERT_OFFSET,
    cassette_nt: int = CASSETTE_LEN,
) -> Path:
    validate_paper_protocol(context_bp, mutation_mode, mutation_offset_nt, cassette_nt)
    if gff_type != "CDS":
        raise ValueError("The paper cohort uses CDS coordinates.")
    stem = f"{model_name}_{phage_name}"
    if step_id is not None:
        stem += f"_step{step_id}"
    return (
        Path(results_dir)
        / phage_name
        / f"{stem}_evo_style_multistopInsert_ctx8192.pkl"
    )


def ensure_output_available(path: str | Path, *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"Result already exists: {path}; pass --force to replace it."
        )


def save_evo_style_results(
    payload: dict, path: str | Path, *, force: bool = False
) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"Result already exists: {path}; pass --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"Saved: {path}")


def load_evo_style_results(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
