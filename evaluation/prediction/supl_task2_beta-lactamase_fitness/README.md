# TEM-1 β-lactamase fitness

Predict experimental fitness from the coding sequences of TEM-1 variants.

The dataset comes from Firnberg et al. (2014),
[A comprehensive, high-resolution map of a gene’s fitness landscape](https://doi.org/10.1093/molbev/msu081).
Codon substitutions are applied to the pBR322 TEM-1 reference CDS (J01749.1).
The benchmark contains 15,167 unique sequences, each 861 nt long.

Frozen model representations are evaluated with a StandardScaler and RidgeCV.
Five-fold KFold uses shuffled splits with seed 42; scaling is fitted within each training fold.
RidgeCV selects among 500 logarithmically spaced penalties from 10⁻⁶ to 10⁶.
The reported metric is the mean Spearman correlation across the five held-out folds.
