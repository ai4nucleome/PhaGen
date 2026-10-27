import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "evaluation/generation/task1_unconditional_generation/scripts/sequence_analysis"
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"common": load_script("common")}):
    blastn = load_script("aggregate_blastn")


def hsp(name, start, end, target="target_1", group="PhaGen", length=100, **overrides):
    span = abs(end - start) + 1
    row = {
        "group": group,
        "seq_name": name,
        "subject_id": target,
        "pident": 95.0,
        "alignment_length": span,
        "qstart": start,
        "qend": end,
        "qlen": length,
        "query_span_bp": span,
        "hsp_query_coverage_pct": 100 * span / length,
        "evalue": 1e-20,
        "bitscore": 100.0,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "coordinates,covered_bp,n_targets",
    [
        ([(1, 10, "a"), (21, 30, "b")], 20, 2),
        ([(1, 20, "a"), (10, 30, "a"), (30, 40, "b"), (41, 50, "c"), (7, 12, "a")], 50, 3),
        ([(100, 81, "a"), (60, 70, "b")], 31, 2),
        ([(1, 100, "a"), (1, 100, "b"), (50, 60, "a")], 100, 2),
        ([(5, 5, "a")], 1, 1),
    ],
)
def test_inclusive_query_union_across_all_targets(coordinates, covered_bp, n_targets):
    rows = [hsp("query", start, end, target) for start, end, target in coordinates]
    queries, summaries = blastn.aggregate(reversed(rows), {"PhaGen": {"query": 100}})
    assert queries == [{
        "group": "PhaGen", "seq_name": "query", "length_bp": 100, "status": "hit",
        "n_targets": n_targets, "n_hsps": len(coordinates), "covered_bp": covered_bp,
        "merged_query_coverage_pct": float(covered_bp),
    }]
    assert summaries[0]["n_with_hit"] == 1
    assert summaries[0]["max_merged_query_coverage_pct"] == covered_bp


def test_all_fasta_queries_and_threshold_include_no_hits_in_deterministic_order():
    lengths = {"megaDNA": {"z_none": 100, "b_above": 100, "a_ten": 100}, "PhaGen": {"q": 200}}
    rows = [hsp("b_above", 1, 11, group="megaDNA"), hsp("a_ten", 100, 91, group="megaDNA")]
    queries, summaries = blastn.aggregate(rows, lengths)
    assert [(row["group"], row["seq_name"]) for row in queries] == [
        ("megaDNA", "z_none"), ("megaDNA", "b_above"), ("megaDNA", "a_ten"), ("PhaGen", "q")
    ]
    assert queries[0] == {
        "group": "megaDNA", "seq_name": "z_none", "length_bp": 100, "status": "no_hit",
        "n_targets": 0, "n_hsps": 0, "covered_bp": 0, "merged_query_coverage_pct": 0.0,
    }
    assert summaries == [
        {"group": "megaDNA", "n_input": 3, "n_with_hit": 2, "n_no_hit": 1,
         "n_coverage_le10": 2, "coverage_le10_pct": 100 * 2 / 3, "max_merged_query_coverage_pct": 11.0},
        {"group": "PhaGen", "n_input": 1, "n_with_hit": 0, "n_no_hit": 1,
         "n_coverage_le10": 1, "coverage_le10_pct": 100.0, "max_merged_query_coverage_pct": 0.0},
    ]


def test_native_eleven_column_input_matches_normalized_input(tmp_path):
    lengths = {blastn.COHORTS[key]: {"query": 2000} for key in blastn.GENERATED}
    for key in blastn.GENERATED:
        path = tmp_path / f"{key}.blastn.tsv"
        path.write_text("")
        path.with_suffix(".tsv.done").write_text("completed\n")
    (tmp_path / "phagen.blastn.tsv").write_text(
        "query\ta\t95\t200\t1\t200\t2000\t99\t1e-20\t100\t0\n"
        "query\tb\t95\t150\t300\t151\t2000\t99\t1e-20\t100\t0\n"
        "query\ta\t95\t51\t501\t451\t2000\t99\t1e-20\t100\t0\n"
    )
    normalized = [hsp("query", start, end, target, length=2000) for start, end, target in [
        (1, 200, "a"), (300, 151, "b"), (501, 451, "a"),
    ]]
    actual = blastn.aggregate(blastn.native_rows(tmp_path), lengths)
    assert actual == blastn.aggregate(normalized, lengths)
    assert actual[0][0]["covered_bp"] == 351
    assert actual[0][0]["merged_query_coverage_pct"] == 17.55
    assert all(row["covered_bp"] == 0 for row in actual[0][1:])


def test_native_input_requires_completion_markers_and_eleven_columns(tmp_path):
    path = tmp_path / "phagen.blastn.tsv"
    path.write_text("query\ttarget\n")
    with pytest.raises(ValueError, match="Completed-search marker missing"):
        list(blastn.native_rows(tmp_path))
    path.with_suffix(".tsv.done").write_text("completed\n")
    with pytest.raises(ValueError, match="eleven-column"):
        list(blastn.native_rows(tmp_path))
    path.write_text("query\ta\t95\t10\t1\t10\t0\t10\t1e-20\t100\t0\n")
    with pytest.raises(ValueError, match="length must be positive"):
        list(blastn.native_rows(tmp_path))


@pytest.mark.parametrize("overrides", [
    {"qstart": 0}, {"qend": 101}, {"qlen": 99}, {"query_span_bp": 11},
    {"alignment_length": 9}, {"hsp_query_coverage_pct": 11.0}, {"pident": float("nan")},
    {"bitscore": float("inf")}, {"evalue": 0.6}, {"seq_name": "absent"}, {"group": "absent"},
])
def test_invalid_alignments_are_rejected(overrides):
    with pytest.raises(ValueError):
        blastn.aggregate([hsp("query", 1, 10, **overrides)], {"PhaGen": {"query": 100}})


def test_rounded_hsp_coverage_is_accepted_without_changing_computed_coverage():
    rows = [hsp("query", 1, 1, length=3, hsp_query_coverage_pct="33.333333")]
    queries, _ = blastn.aggregate(rows, {"PhaGen": {"query": 3}})
    assert queries[0]["covered_bp"] == 1
    assert queries[0]["merged_query_coverage_pct"] == 100 / 3


def test_cli_writes_two_tables_for_a_completed_search_with_no_hsps(tmp_path):
    sequences, output = tmp_path / "sequences", tmp_path / "results"
    sequences.mkdir()
    for key in blastn.GENERATED:
        (sequences / f"{key}.fasta").write_text(f">z\n{'A' * 2000}\n>a\n{'C' * 2000}\n")
    hsps = tmp_path / "empty.hsps.tsv"
    hsps.write_text("\t".join(blastn.HSP_FIELDS) + "\n")
    command = [sys.executable, str(SCRIPT_DIR / "aggregate_blastn.py"), "--sequence-dir", str(sequences),
               "--hsps", str(hsps), "--output-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert sorted(path.name for path in output.iterdir()) == ["queries.tsv", "summary.tsv"]
    with (output / "queries.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert reader.fieldnames == blastn.QUERY_FIELDS
        queries = list(reader)
    assert [(row["group"], row["seq_name"]) for row in queries] == [
        (blastn.COHORTS[key], name) for key in blastn.GENERATED for name in ("z", "a")
    ]
    assert all(row["status"] == "no_hit" and float(row["merged_query_coverage_pct"]) == 0 for row in queries)
    with (output / "summary.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert reader.fieldnames == blastn.SUMMARY_FIELDS
        assert all(row["n_coverage_le10"] == "2" and float(row["coverage_le10_pct"]) == 100 for row in reader)
    assert subprocess.run(command, capture_output=True).returncode != 0
    subprocess.run(command + ["--force"], check=True, capture_output=True)
