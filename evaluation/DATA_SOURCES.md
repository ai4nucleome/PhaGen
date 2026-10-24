# Data sources and upstream terms

This document records the upstream experimental data and reference sequences used
in the PhaGen evaluation materials. Source datasets retain their applicable
upstream terms. The project's software license does not relicense third-party
biological data; software attribution is described in [NOTICE.md](../NOTICE.md).
This document does not assign a new license to project-authored predictions,
statistical summaries or figures.

## Experimental benchmarks

| Benchmark | Source and input used | Upstream terms |
| --- | --- | --- |
| infA fitness | [Kelsic et al. (2016)](https://doi.org/10.1016/j.cels.2016.11.004), `fig1f-data.csv` in the publication's `mmc5.zip` supplement. Mutant CDSs are reconstructed from [NC_000913.3](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3), complement(926225..926443); repeated WT entries are collapsed. | The [publisher's article page](https://www.sciencedirect.com/science/article/pii/S2405471216303684) identifies an [Elsevier user license](https://www.elsevier.com/about/policies-and-standards/open-access-licenses/elsevier-user). A separate data license for the source CSV has not been identified. The article license is recorded with that scope; public access to the supplement is not treated as an additional redistribution grant. |
| TEM-1 beta-lactamase fitness | [Firnberg et al. (2014)](https://doi.org/10.1093/molbev/msu081), Supplementary Data S1, workbook sheet `S1 Codon fitnesses`. Mutant CDSs are reconstructed from [J01749.1](https://www.ncbi.nlm.nih.gov/nuccore/J01749.1), complement(3293..4153). | The [publisher](https://academic.oup.com/mbe/article/31/6/1581/2925654) licenses the article under [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/). No separate workbook license has been identified. Preserve the source attribution and applicable upstream terms; the source measurements are not relicensed under Apache-2.0. |
| Gene essentiality | Curated lambda and P1 annotations derived from [Piya et al. (2023)](https://doi.org/10.1371/journal.pbio.3002416), including [Table 1](https://doi.org/10.1371/journal.pbio.3002416.t001) and [Table 2](https://doi.org/10.1371/journal.pbio.3002416.t002). Reference genomes and CDS annotations are [NC_001416.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_001416.1) and [NC_005856.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_005856.1). | The article and its published tables are [CC BY 4.0](https://journals.plos.org/plosbiology/s/licenses-and-copyright). Cite the source and identify the curated annotations as derived material. NCBI sequence records follow the separate policy below. |
| T7 SNP mutability | [Robins, Faruque and Mekalanos (2013)](https://doi.org/10.1073/pnas.1222538110), supplementary Mut-seq workbook `sd01.xlsx`. The retained experimental input is a curated version of this supplement; mutant CDSs use reference [V01146.1](https://www.ncbi.nlm.nih.gov/nuccore/V01146.1). | The [publisher's supporting-information page](https://www.pnas.org/doi/10.1073/pnas.1222538110) identifies the workbook. An explicit dataset-level reuse license has not been confirmed. Free access to the article is not recorded here as a CC BY grant for the workbook. |
| Phage promoter recognition | [Sampaio et al. (2019)](https://doi.org/10.1093/bioinformatics/btz580), [PhagePromoter author data repository](https://github.com/martaS95/PhagePromoter/tree/master/Data): `DATASET2400.csv`, `promoters.fasta` and `list_phages.txt`. These annotations and nucleotide reference records are used to construct the 65-bp benchmark windows. | An explicit dataset-level reuse license has not been identified in the linked data repository. Retain the study citation and source links. Reference-genome provenance is retained separately in the FASTA headers. |
| Virion-protein subtypes | [Shang et al. (2023), PhaVIP](https://doi.org/10.1093/bioinformatics/btad229). The [author download service](https://phage.ee.cityu.edu.hk/download) provides `dataset_by_similarity.zip`; this benchmark uses `train_40.csv` and `test_40.csv`, with NCBI CDS mappings. | An explicit data license has not been confirmed for the separately distributed archive. The PhaVIP software's GPL-3.0 license is not taken as evidence of a license for this external data archive. NCBI CDS mappings follow the policy below. |
| Tail-protein CDS host prediction | [Du et al. (2026)](https://doi.org/10.1038/s41467-026-70613-x), dataset [10.5281/zenodo.17340915](https://doi.org/10.5281/zenodo.17340915): `data.zip`, member `data/benchmark/DRTail.csv`. The benchmark retains the source CDSs and host labels after the documented preprocessing. | The [dataset record](https://zenodo.org/records/17340915) explicitly declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the dataset and study, preserve attribution, and indicate preprocessing. This is the dataset's declaration, independently of the article or software license. |

## Natural reference sequences and training-corpus provenance

The generation reference contains 10,000 sequences: 5,000 from the PhaGen training
partition and 5,000 from OpenGenome stage 2. Training-corpus provenance follows
[Shao and Yan (2024), megaDNA](https://doi.org/10.1038/s41467-024-53759-4) and its
[training sequence identifiers](https://github.com/lingxusb/megaDNA/blob/main/training/training_seqs.tsv).
The retained `generation/task1_unconditional_generation/data/cohorts/natural_records.tsv`
records the selected sources and identifiers.

| Source collection | Input identity and retained reference records | Upstream terms |
| --- | --- | --- |
| Metagenomic Gut Virus catalogue (MGV) | [Nayfach et al. (2021)](https://doi.org/10.1038/s41564-021-00928-6), release [MGV_v1.0_2021_07_08](https://portal.nersc.gov/MGV/MGV_v1.0_2021_07_08/). The Natural reference retains 2,627 MGV records. | The release [data usage policy](https://portal.nersc.gov/MGV/MGV_v1.0_2021_07_08/README.txt) permits use without additional restrictions and requests citation of the study. It does not name a standard license identifier. |
| Gut Phage Database (GPD) | [Camarillo-Guerrero et al. (2021)](https://doi.org/10.1016/j.cell.2021.01.029); [Sanger resource](https://www.sanger.ac.uk/data/gut-phage-database/) and [EBI sequence/metadata distribution](https://ftp.ebi.ac.uk/pub/databases/metagenomics/genome_sets/gut_phage_database/). The Natural reference retains 1,631 GPD records. | Follow the linked resource's data-use terms. The [distribution README](https://ftp.ebi.ac.uk/pub/databases/metagenomics/genome_sets/gut_phage_database/README.txt) does not state a separate standardized data license. The analysis-code repository's MIT license is not used to license the externally distributed nucleotide data. |
| NCBI nucleotide resources | GenBank/RefSeq accession versions are retained in the source records. The Natural reference includes 742 records from the NCBI branch of the training corpus. | NCBI Molecular Data Usage policy; see below. |
| OpenGenome stage 2 | [LongSafari/open-genome](https://huggingface.co/datasets/LongSafari/open-genome), `stage2/train_stage2.parquet`, associated with [Nguyen et al. (2024)](https://doi.org/10.1126/science.ado9336). The Natural reference includes 5,000 sequences, identified by packed-file row and segment. | The upstream [dataset card](https://huggingface.co/datasets/LongSafari/open-genome/blob/main/README.md) declares Apache-2.0 for this dataset. This declaration is recorded for the OpenGenome source; it is not inferred from the PhaGen code license. |

The S11 training-similarity analysis uses the same training-corpus lineage, with
2,000 background queries and the lambda/P1 reference controls. The retained
length and similarity tables are analysis outputs; the complete training FASTA
and BLAST database are external inputs.

## NCBI reference records

The [NCBI Molecular Data Usage policy](https://www.ncbi.nlm.nih.gov/home/about/policies/#molecular-data-usage)
states that NCBI itself imposes no restrictions on use or distribution of its
molecular data. It also explains that original submitters may hold rights that
NCBI cannot transfer. Accordingly, the NCBI policy is recorded here without
assigning a blanket CC0 license to GenBank/RefSeq records.

Retain accession versions, source attribution and any record-specific notices
when reusing the reference genomes, annotations or mapped CDSs. Accession
versions in FASTA headers and mapping tables take precedence over filenames
that omit the version suffix.

## Attribution and license scope

When using a processed benchmark, cite its original experimental study and
retain the relevant source identifier. The preprocessing scripts and task
READMEs describe the reconstruction, filtering, window selection or mapping
applied for PhaGen. A processed table's upstream measurements and sequences
retain their applicable source terms.

A statement that no separate dataset license was identified describes the
available source documentation; it is not a new permission to redistribute
that material. Article, software and dataset licenses are listed with their
respective scopes. No single code license is applied to the entire collection.
