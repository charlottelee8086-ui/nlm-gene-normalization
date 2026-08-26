# Data preparation

This directory contains the shared data preparation code used by several experiments.

## `split_train_dev.py`

Creates the document-level training and development split used during the project.

The original NLM-Gene training set contains 450 documents. It was split into 360 training documents and 90 development documents. Documents, rather than individual mentions, were used as the split unit.

## `build_contexts.py`

Builds the different text contexts used in the candidate-selection experiments, including sentence-level context, a local character window, and full-abstract context.

## `build_ncbi_symbol_taxid_kb.py`

Builds a lookup table from NCBI Gene data. The table links gene symbols and synonyms to GeneIDs and TaxIDs.

It is used by the species-aware symbol-grounding experiment.

