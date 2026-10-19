# T7 SNP mutability

Predict normalized mutability indices (NMI) for SNPs in 59 T7 phage genes.
Curated T7 Mut-seq measurements are from Robins, Faruque and Mekalanos (2013),
“Coupling mutagenesis and parallel deep sequencing to probe essential residues in a genome or gene.”
Mutant CDSs are reconstructed against the V01146.1 reference genome.
Targets are the mean of available replicate-A and replicate-B NMI measurements.
The benchmark contains 20,087 mutation records across 59 genes.

Frozen model representations are evaluated separately for each gene.
Each probe uses StandardScaler and RidgeCV with five shuffled folds (seed 42).
Ridge penalties span 10⁻⁶ to 10⁶ on a 500-point logarithmic grid.
The main metric averages signed fold Spearman correlations within genes, then across genes.
The supplementary comparison retains the 46 genes with at least 100 records.
