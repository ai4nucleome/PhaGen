# Tail-protein CDS host prediction

Predict seven host species from bacteriophage tail-protein coding sequences
in the DRTail benchmark of Du et al. (2026).

After excluding non-triplet CDSs and records without identifiable source genomes,
the benchmark contains 12,236 sequences from 2,214 phage genomes.

Twelve frozen DNA models provide sequence representations.
All models share StratifiedGroupKFold splits with five folds and seed 42,
grouped by source genome, with no genome, identical-DNA or identical-protein overlap.
Each training fold fits StandardScaler and a class-balanced logistic regression
with C=1 and max_iter=5,000. The primary metric is mean held-out-fold macro-F1;
MCC and accuracy are supplementary metrics, with sample standard deviations across folds.
