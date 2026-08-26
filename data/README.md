# Data

Large datasets and generated model files are not stored directly in this repository.

## NLM-Gene

The experiments use the NLM-Gene benchmark.

The original training set contains 450 documents. For development experiments, it was split at the document level into:

- 360 training documents
- 90 development documents

The official 100-document test set was kept unchanged.

All mentions from the same document were kept in the same split.

## NCBI Gene

NCBI Gene `gene_info` data was used to build symbol, synonym, GeneID, and species lookup tables.

The large source file and generated full dictionaries are not committed to Git. Scripts for building the required resources are provided under `src/data/`.

## Generated files

Candidate lists, complete prompt files, model predictions, SapBERT embeddings, and model checkpoints can be regenerated from the scripts in this repository and are not stored here by default.

Small split files, examples, and final result tables may be included when they are useful for reproducibility.
