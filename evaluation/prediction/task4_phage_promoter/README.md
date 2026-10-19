# Phage promoter recognition

This benchmark uses PhagePromoter annotations to classify 65 bp genomic windows:
800 promoters and 2,400 accession-matched, GC-matched non-promoter windows.
Negative windows avoid annotated promoters. Genomes are grouped by accession,
merging accessions linked by positive-promoter similarity of at least 95% on
either strand, yielding 36 groups and five stratified group folds (seed 11).
Frozen model embeddings feed a class-balanced logistic regression classifier
with training-fold standardization (C=1, liblinear). Performance is reported as
five-fold mean AUROC, trapezoidal AUPRC, and MCC at a probability threshold of 0.5.
