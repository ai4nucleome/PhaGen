# Genome-scale sequence generation

PhaGen, megaDNA, Evo1-7B and Evo2-7B each generate 1,024 samples with a
131,072-token budget. Model-native boundaries are used to extract ACGT sequences
of at least 2,000 bp. PhaGen uses 256 entropy-based
denoising steps with temperature 1.1, top-p 0.9, top-k 4 and selection temperature 0.1.

The reference comprises 10,000 Duplodnaviria sequences, sampled equally from the
PhaGen training corpus and OpenGenome stage 2, with composition-preserving shuffled
controls. Evaluations cover viral classification, sequence and ORF statistics,
nucleotide similarity, protein function and composition, generation time,
and the decoding comparisons reported in the supplementary figures.

## BLASTN merged query coverage

The current Fig. S5 uses every generated sequence, including queries without
reported hits at zero coverage. For each query, merge the inclusive query
intervals of all reported BLASTN HSPs across all targets, count overlapping bases
once, and divide by the complete query length. The saved searches use megablast,
E-value <= 0.5 and up to 100 targets per query.

The current query table, summary, standalone plotting script, and PDF/PNG figure
are included in this repository. From the repository root, redraw the figure
from the included query table into a separate directory:

```bash
python evaluation/generation/task1_unconditional_generation/scripts/sequence_analysis/plot_blastn_coverage.py --output-dir outputs/blastn_coverage
```

To recompute the query table as well, obtain the FASTA cohorts and
`results/blastn/core_nt.hsps.tsv` from the complete Zenodo archive, then run:

```bash
python evaluation/generation/task1_unconditional_generation/scripts/sequence_analysis/aggregate_blastn.py --force
python evaluation/generation/task1_unconditional_generation/scripts/sequence_analysis/plot_blastn_coverage.py --force
```

Use `--force` to replace existing outputs, or `--output-dir` to write to another
directory. For fresh native searches, pass `--native-dir` to the aggregation
script; all four search outputs must have their `.done` completion markers.
No new BLAST search is needed to aggregate the saved HSP table.

The aggregation writes `results/blastn/queries.tsv` (one row per query, with
`covered_bp` and `merged_query_coverage_pct`) and `results/blastn/summary.tsv`.
The paper data contain 6,058 queries, including 4,596 without reported hits;
6,015/6,058 (99.3%) have merged coverage at or below 10%.

The figure is written to `figs/blastn_merged_query_coverage.pdf` and `.png`,
with input SHA256 and plotting settings in
`figs/blastn_merged_query_coverage.metadata.json`. It keeps all queries, a 10%
reference line, and a vertical axis that is linear near zero and logarithmic
above 0.01%. These outputs replace the earlier best-HSP table and figure;
single-HSP coverage is a different statistic and is not the current Fig. S5.
The Zenodo archive predates this correction; use the BLASTN aggregation and
plotting scripts and corrected outputs in this repository for the current figure.
