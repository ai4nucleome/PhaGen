# Phage gene essentiality

Evaluate zero-shot prediction of essential genes in bacteriophages lambda and P1.
Essentiality annotations follow [Piya et al. (2023)](https://doi.org/10.1371/journal.pbio.3002416).
Reference genomes and CDS annotations are NC_001416.1 and NC_005856.1.
The evaluated CDSs comprise 30 essential and 37 non-essential lambda genes,
and 27 essential and 80 non-essential P1 genes; untested labels are excluded.

A 15-nt stop cassette, `TAATAATAATAGTGA`, is inserted after the first 12 coding nucleotides.
Wild-type and perturbed CDSs are separately extended with genomic flanks to 8,192 nt.
PhaGen and NTv3 use masked-region scores; megaDNA and Evo use native mean log-likelihoods.
The essentiality score is the wild-type score minus the perturbed score.
Performance is measured using AUROC and trapezoidal AUPRC, with essential genes positive.

Training-corpus similarity uses summed BLAST alignment coverage and 2,000 sampled training genomes.
