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
