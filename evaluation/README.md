# PhaGen Evaluation

Evaluation code for genome-scale DNA generation and seven sequence-understanding benchmarks.

Data, formal results, plotting code, and figures are available on
[Zenodo](https://zenodo.org/records/23477083).
Download `PhaGen-complete.zip` for the complete materials, which preserve the
evaluation directory layout. The archive includes the PhaGen checkpoint under
`PhaGen/models/PhaGen/`. Baseline model weights and full external reference
databases are obtained separately.

## Generation

[Genome-scale generation](generation/task1_unconditional_generation/):
DNA generation with PhaGen, megaDNA, Evo1-7B, and Evo2-7B, including sequence
and protein analyses, decoding comparisons, and generation-time measurements.

## Prediction

| Task | Directory |
| --- | --- |
| Lambda and P1 gene essentiality | [task1_gene_enssentiality](prediction/task1_gene_enssentiality/) |
| T7 SNP mutability | [task2_proteins_functions](prediction/task2_proteins_functions/) |
| Phage promoter recognition | [task4_phage_promoter](prediction/task4_phage_promoter/) |
| Virion-protein subtype prediction | [task5_pvp](prediction/task5_pvp/) |
| Tail-protein CDS host prediction | [task6_tail-based_host](prediction/task6_tail-based_host/) |
| infA mutation fitness | [supl_task1_infA_fitness](prediction/supl_task1_infA_fitness/) |
| TEM-1 β-lactamase fitness | [supl_task2_beta-lactamase_fitness](prediction/supl_task2_beta-lactamase_fitness/) |

See each task's README for the benchmark description and evaluation protocol.
Data-source citations and applicable upstream terms are included in the Zenodo archive.
