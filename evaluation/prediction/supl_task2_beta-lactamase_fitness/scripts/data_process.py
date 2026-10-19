from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = TASK_DIR / "data" / "original"
DEFAULT_OUT_DIR = TASK_DIR / "data" / "processed"

RAW_USECOLS = [0, 1, 2, 3, 4, 5, 19, 20, 21]

FITNESS_COLUMNS = [
    "Ambler_Position",
    "WT_codon",
    "Mutant_codon",
    "WT_AA",
    "Mutant_AA",
    "Base_Changes",
    "Total_Counts",
    "Fitness",
    "Error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TEM-1 CDS and build the Firnberg codon-fitness evaluation dataset."
    )
    parser.add_argument("--fasta", type=Path, default=RAW_DATA_DIR / "sequence.fasta")
    parser.add_argument("--gff3", type=Path, default=RAW_DATA_DIR / "sequence.gff3")
    parser.add_argument(
        "--fitness",
        type=Path,
        default=RAW_DATA_DIR / "Firnberg_beta-lactamase.xlsx",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def clean_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def translate_codon(codon: str) -> str:
    return str(Seq(codon).translate(table=11))


def parse_gff3_for_tem1(gff3_path: Path) -> tuple[int, int, str]:
    with gff3_path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "CDS":
                continue
            attrs = cols[8]
            if "gene=bla" not in attrs and "beta-lactamase" not in attrs.lower():
                continue
            gff_start = int(cols[3])
            gff_end = int(cols[4])
            strand = cols[6]
            py_start = gff_start - 1
            py_end = gff_end
            print(f"  [GFF3] 找到 TEM-1 CDS: {cols[0]}:{gff_start}-{gff_end} ({strand} strand)")
            print(f"         Python 切片: [{py_start}:{py_end}]，长度 {py_end - py_start} bp")
            return py_start, py_end, strand
    raise ValueError(f"GFF3 中未找到 TEM-1 (bla) CDS feature: {gff3_path}")


def extract_wt_cds(fasta_path: Path, py_start: int, py_end: int, strand: str) -> tuple[str, str]:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if len(records) != 1:
        raise ValueError(f"{fasta_path} should contain exactly one FASTA record, got {len(records)}.")
    record = records[0]
    print(f"  [FASTA] 读取序列: {record.id}，长度 {len(record.seq)} bp")

    region = record.seq[py_start:py_end]
    cds_seq = region.reverse_complement() if strand == "-" else region
    if strand == "-":
        print("  [提取] 负链，已取反向互补")
    else:
        print("  [提取] 正链，直接截取")

    wt_cds = str(cds_seq).upper()
    print(f"  [提取] CDS 长度: {len(wt_cds)} bp")

    stop_codon = wt_cds[-3:]
    protein = str(Seq(wt_cds[:-3]).translate(table=11))
    if len(wt_cds) != 861:
        raise ValueError(f"CDS 长度应为 861 bp，实际为 {len(wt_cds)} bp")
    if not protein.startswith("MSIQHFR"):
        raise ValueError(f"蛋白质起始序列错误: {protein[:7]}")
    if stop_codon not in {"TAA", "TAG", "TGA"}:
        raise ValueError(f"终止密码子异常: {stop_codon}")
    if len(protein) != 286:
        raise ValueError(f"蛋白质长度应为 286 aa，实际为 {len(protein)} aa")

    print(f"  [验证] ✓ 翻译正确: {protein[:10]}...{protein[-10:]} ({len(protein)} aa)")
    print(f"  [验证] ✓ 终止密码子: {stop_codon}")
    return record.id, wt_cds


def load_fitness_table(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        excel_path,
        sheet_name="S1 Codon fitnesses",
        header=0,
        skiprows=1,
        usecols=RAW_USECOLS,
    )
    if df.shape[1] != len(FITNESS_COLUMNS):
        raise ValueError(f"Unexpected Excel layout in {excel_path}: expected {len(FITNESS_COLUMNS)} selected columns.")
    df.columns = FITNESS_COLUMNS

    df["Ambler_Position"] = pd.to_numeric(df["Ambler_Position"], errors="coerce").astype("Int64")
    df["WT_codon"] = clean_string_series(df["WT_codon"])
    df["Mutant_codon"] = clean_string_series(df["Mutant_codon"])
    df["WT_AA"] = clean_string_series(df["WT_AA"])
    df["Mutant_AA"] = clean_string_series(df["Mutant_AA"])
    df["Base_Changes"] = pd.to_numeric(df["Base_Changes"], errors="coerce").astype("Int64")
    df["Total_Counts"] = pd.to_numeric(df["Total_Counts"], errors="coerce").astype("Int64")
    df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")

    df = df[df["Ambler_Position"].notna()].copy()
    invalid_codons = (~df["WT_codon"].str.fullmatch(r"[ACGT]{3}", na=False)).sum() + (
        ~df["Mutant_codon"].str.fullmatch(r"[ACGT]{3}", na=False)
    ).sum()
    if invalid_codons:
        raise ValueError(f"Found {invalid_codons} invalid codon strings in {excel_path}.")

    return df


def build_ambler_mapping(df_s1: pd.DataFrame) -> tuple[dict[int, int], pd.DataFrame]:
    wt_pos = (
        df_s1
        .drop_duplicates("Ambler_Position")[["Ambler_Position", "WT_AA", "WT_codon"]]
        .sort_values("Ambler_Position")
        .reset_index(drop=True)
    )
    wt_pos["aa_index"] = wt_pos.index

    ambler_vals = wt_pos["Ambler_Position"].to_numpy(dtype=int)
    gaps = [
        (int(ambler_vals[i]), int(ambler_vals[i + 1]))
        for i in range(len(ambler_vals) - 1)
        if ambler_vals[i + 1] - ambler_vals[i] > 1
    ]
    if gaps:
        print(f"  [Ambler] 发现编号 gap（class A beta-lactamase 标准）: {gaps}")

    return dict(zip(wt_pos["Ambler_Position"].astype(int), wt_pos["aa_index"].astype(int))), wt_pos


def mutation_type_from_aas(wt_aa: str, mutant_aa: str) -> str:
    if mutant_aa == "*" and wt_aa != "*":
        return "Premature stop"
    if wt_aa == mutant_aa:
        return "Synonymous"
    return "Nonsynonymous"


def build_dataset(df_s1: pd.DataFrame, wt_cds: str, ambler_to_idx: dict[int, int]) -> pd.DataFrame:
    df_valid = df_s1[df_s1["Fitness"].notna()].copy()
    rows: list[dict[str, object]] = []

    for row in df_valid.itertuples(index=False):
        ambler_position = int(row.Ambler_Position)
        aa_index = ambler_to_idx[ambler_position]
        wt_codon = str(row.WT_codon)
        mutant_codon = str(row.Mutant_codon)
        wt_aa = str(row.WT_AA)
        mutant_aa = str(row.Mutant_AA)
        base_changes = int(row.Base_Changes)

        codon_start = aa_index * 3
        codon_end = codon_start + 3
        wt_codon_in_seq = wt_cds[codon_start:codon_end]
        if wt_codon_in_seq != wt_codon:
            raise ValueError(
                f"密码子不匹配 Ambler {ambler_position}: 序列={wt_codon_in_seq}, 数据={wt_codon}"
            )

        translated_wt = translate_codon(wt_codon)
        translated_mut = translate_codon(mutant_codon)
        if translated_wt != wt_aa:
            raise ValueError(
                f"WT codon translation mismatch at Ambler {ambler_position}: {wt_codon}->{translated_wt}, expected {wt_aa}"
            )
        if translated_mut != mutant_aa:
            raise ValueError(
                f"Mutant codon translation mismatch at Ambler {ambler_position}: {mutant_codon}->{translated_mut}, expected {mutant_aa}"
            )

        computed_base_changes = sum(a != b for a, b in zip(wt_codon, mutant_codon))
        if computed_base_changes != base_changes:
            raise ValueError(
                f"Base_Changes mismatch at Ambler {ambler_position}: file={base_changes}, computed={computed_base_changes}"
            )

        mutant_cds = wt_cds[:codon_start] + mutant_codon + wt_cds[codon_end:]
        rows.append(
            {
                "ambler_position": ambler_position,
                "mutation": f"{wt_codon}_{codon_start+1}-{codon_end}_{mutant_codon}",
                "mutation_type": mutation_type_from_aas(wt_aa, mutant_aa),
                "fitness": float(row.Fitness),
                "fitness_error": float(row.Error) if pd.notna(row.Error) else np.nan,
                "seq": mutant_cds,
            }
        )

    df_out = pd.DataFrame(rows).sort_values(
        ["ambler_position"], ignore_index=True
    )
    if df_out["mutation"].duplicated().any():
        raise ValueError("Duplicate mutation_codon entries detected in processed dataset.")
    return df_out


def save_fasta(path: Path, record_id: str, seq: str) -> None:
    with path.open("w") as handle:
        handle.write(f">{record_id}\n")
        for i in range(0, len(seq), 60):
            handle.write(seq[i : i + 60] + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: 解析 GFF3 ===")
    py_start, py_end, strand = parse_gff3_for_tem1(args.gff3)

    print("\n=== Step 2: 提取并验证 WT CDS ===")
    accession, wt_cds = extract_wt_cds(args.fasta, py_start, py_end, strand)

    wt_fasta_path = args.out_dir / "TEM1_WT_CDS.fasta"
    save_fasta(
        wt_fasta_path,
        "TEM1_WT_CDS | J01749.1 (pBR322) | 861 bp | encodes UniProt P62593",
        wt_cds,
    )
    print(f"  [保存] WT CDS → {wt_fasta_path}")

    print("\n=== Step 3: 读取适应度数据 (S1 Codon fitnesses) ===")
    df_s1 = load_fitness_table(args.fitness)
    print(f"  [读取] 总行数: {len(df_s1):,}")
    print(f"  [读取] 有 fitness 的行数: {df_s1['Fitness'].notna().sum():,}")

    print("\n=== Step 4: 构建 Ambler → aa_index 映射 ===")
    ambler_to_idx, _ = build_ambler_mapping(df_s1)

    print("  [验证] 抽查活性位点密码子:")
    for ambler, expected_aa in [(70, "S"), (73, "K"), (130, "S"), (166, "E")]:
        idx = ambler_to_idx[ambler]
        codon = wt_cds[idx * 3 : idx * 3 + 3]
        aa = translate_codon(codon)
        status = "✓" if aa == expected_aa else "✗"
        print(
            f"    Ambler {ambler:3d} → aa_index {idx:3d} → codon {codon} → {aa} "
            f"(expected {expected_aa}) {status}"
        )

    print("\n=== Step 5: 构建突变 CDS 数据集 ===")
    df_out = build_dataset(df_s1, wt_cds, ambler_to_idx)
    compact_path = args.out_dir / "mutants_sequences.tsv"

    df_out[
        [
            "ambler_position",
            "seq",
            "fitness",
            "mutation",
            "mutation_type",
            "fitness_error",
        ]
    ].to_csv(
        compact_path, sep="\t", index=False
    )

    print("\n=== 完成 ===")
    print(f"  主数据文件:   {compact_path}")
    print(f"    总样本数:   {len(df_out):,}")
    print(f"    fitness 范围: {df_out['fitness'].min():.4f} – {df_out['fitness'].max():.4f}")
    print(f"  WT CDS FASTA: {wt_fasta_path}")


if __name__ == "__main__":
    main()
