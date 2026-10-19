#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


TASK_DIR = Path(__file__).resolve().parent.parent
MODELS = {
    "phagen": "PhaGen",
    "megadna": "megaDNA",
    "ntv3_8m": "NTv3-8M",
    "ntv3_100m": "NTv3-100M",
    "ntv3_650m": "NTv3-650M",
    "evo": "Evo1-7B",
    "evo2_1b": "Evo2-1B",
    "evo2": "Evo2-7B",
}
MODEL_NAMES = {
    "phagen": "phagen",
    "megadna": "megadna",
    "NTv3_8M_pre": "ntv3_8m",
    "NTv3_100M_pre": "ntv3_100m",
    "NTv3_650M_pre": "ntv3_650m",
    "evo-1-131k-base": "evo",
    "evo2_1b_base": "evo2_1b",
    "evo2_7b": "evo2",
}
PHAGES = {
    "lambda": ("NC_001416.1", 30, 37),
    "p1": ("NC_005856.1", 27, 80),
}
SUMMARY_FIELDS = [
    "model_key", "model", "phage", "context_bp", "score_type",
    "AUROC", "AUPRC", "n_E", "n_NE", "result_file",
]


def extract_scores_and_labels(genes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    
    labelled = [gene for gene in genes if gene.get("label") is not None]
    scores = np.asarray([gene["delta"] for gene in labelled], dtype=float)
    labels = np.asarray([gene["label"] for gene in labelled], dtype=int)
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be binary.")
    return scores, labels


def compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall[::-1], precision[::-1]))


def evaluate_result(path: Path, data_dir: Path) -> dict:
    from t12_evo_style_utils import load_gene_annotations

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    phage = payload["phage_name"]
    if phage not in PHAGES:
        raise ValueError(f"Unknown phage in {path}.")
    if payload.get("context_bp") != 8192:
        raise ValueError(f"Expected 8,192-bp context in {path}.")
    if payload.get("mutation_mode") != "multistop12_insert":
        raise ValueError(f"Expected the stop-cassette insertion protocol in {path}.")
    for field, expected in {"gff_type": "CDS", "mutation_offset_nt": 12, "cassette_len_nt": 15}.items():
        if payload.get(field) != expected:
            raise ValueError(f"Expected {field}={expected!r} in {path}.")
    model_key = payload.get("model_key") or MODEL_NAMES.get(payload.get("model_name"))
    if model_key not in MODELS:
        raise ValueError(f"Unknown paper model in {path}.")
    expected_score_type = (
        "masked_marginals_cassette" if model_key == "phagen"
        else "ntv3_masked_marginals_cassette" if model_key.startswith("ntv3_")
        else "mean_log_likelihood"
    )
    if payload.get("score_type") != expected_score_type:
        raise ValueError(f"Unexpected scoring procedure in {path}.")
    if model_key == "phagen" and (
        str(payload.get("step_id")) != "12800" or payload.get("pad_attention_mode") != "attend_pad"
    ):
        raise ValueError(f"Expected the step-12800 attend_pad PhaGen experiment in {path}.")
    accession, expected_e, expected_ne = PHAGES[phage]
    annotations = load_gene_annotations(
        str(data_dir / phage / f"{accession}.gff3"),
        str(data_dir / phage / f"{phage}_gene_essentiality_CORRECTED.csv"),
        gff_type="CDS",
    )
    labels_by_tag = {
        gene["locus_tag"]: gene["label"]
        for gene in annotations if gene["label"] is not None
    }
    genes = [dict(gene) for gene in payload["genes"]]
    tags = [gene["locus_tag"] for gene in genes]
    if len(tags) != len(set(tags)):
        raise ValueError(f"Duplicate CDS locus tags in {path}.")
    for gene in genes:
        gene["label"] = labels_by_tag.get(gene["locus_tag"])
    observed_tags = {gene["locus_tag"] for gene in genes if gene["label"] is not None}
    if observed_tags != set(labels_by_tag):
        raise ValueError(f"Result does not cover the full labelled CDS cohort in {path}.")
    scores, labels = extract_scores_and_labels(genes)
    n_e, n_ne = int(labels.sum()), int((1 - labels).sum())
    if (n_e, n_ne) != (expected_e, expected_ne):
        raise ValueError(f"Unexpected essential/non-essential CDS counts in {path}.")
    try:
        result_file = str(path.resolve().relative_to(TASK_DIR))
    except ValueError:
        result_file = str(path.resolve())
    return {
        "model_key": model_key,
        "model": MODELS[model_key],
        "phage": phage,
        "context_bp": 8192,
        "score_type": payload["score_type"],
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": compute_auprc(scores, labels),
        "n_E": n_e,
        "n_NE": n_ne,
        "result_file": result_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate stop-cassette gene-essentiality scores with AUROC and trapezoidal AUPRC.')
    parser.add_argument("pkl", nargs="*", type=Path, help="Local result pickle files.")
    parser.add_argument("--data-dir", type=Path, default=TASK_DIR / "data")
    parser.add_argument("--results-dir", type=Path, default=TASK_DIR / "results")
    parser.add_argument("--output", type=Path, help="Write a metrics summary CSV.")
    parser.add_argument("--save", action="store_true", help="Write a metrics JSON beside each input.")
    args = parser.parse_args()
    paths = args.pkl or sorted(args.results_dir.glob("*/*.pkl"))
    if not paths:
        parser.error("No result files found.")
    evaluations = [(path, evaluate_result(path, args.data_dir)) for path in paths]
    rows = [row for _, row in evaluations]
    identities = [(row["phage"], row["model_key"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Multiple results supplied for the same model and phage.")
    if not args.pkl:
        expected = {(phage, model) for phage in PHAGES for model in MODELS}
        if set(identities) != expected:
            raise ValueError("The paper comparison requires eight models for each phage.")
    rows.sort(key=lambda row: (list(PHAGES).index(row["phage"]), list(MODELS).index(row["model_key"])))
    print("phage\tmodel\tAUROC\tAUPRC\tn_E\tn_NE")
    for row in rows:
        print(f"{row['phage']}\t{row['model']}\t{row['AUROC']:.10f}\t{row['AUPRC']:.10f}\t{row['n_E']}\t{row['n_NE']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    if args.save:
        for path, row in evaluations:
            path.with_suffix(".evo_metrics.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
