# Virion-protein subtype prediction

Predict seven virion-protein subtypes using the official similarity-40 partitions
from [PhaVIP/PhaBOX](https://phage.ee.cityu.edu.hk/download) (Shang et al., 2023).
Protein accessions are mapped to complete, unambiguous NCBI CDSs of at most 8,192 nt.
After removing conflicting identical-protein groups and duplicate CDSs,
the benchmark contains 14,901 sequences: 7,822 training and 7,079 test records.

Twelve frozen DNA models provide sequence representations.
StandardScaler is fitted on training embeddings only, followed by class-balanced
logistic regression with C=1, lbfgs, max_iter=5,000 and seed 42.
The primary test metric is macro-F1; MCC and accuracy are supplementary metrics.
