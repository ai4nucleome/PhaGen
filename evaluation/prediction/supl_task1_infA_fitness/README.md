# infA mutation fitness

Predict *E. coli* infA fitness from frozen DNA-model representations.
The benchmark contains **4,600 unique 219-nt sequences**: 4,599 mutants and one WT.
Measurements come from [Kelsic et al.](https://doi.org/10.1016/j.cels.2016.11.004)
(`mmc5.zip`, `fig1f-data.csv`); the WT CDS is positions 926225–926443 of
[NC_000913.3](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3).

The probe uses `StandardScaler` followed by `RidgeCV` with 500 alphas from
`1e-6` to `1e6` (internal leave-one-out selection). Shuffled five-fold `KFold`
uses seed 42. The primary metric is mean held-out-fold Spearman correlation.
The sequence models are not fine-tuned.
