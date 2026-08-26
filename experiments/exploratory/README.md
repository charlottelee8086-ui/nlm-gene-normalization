# Exploratory experiments

This directory contains experiments that were useful during development but were not part of the main report comparison.

They are kept separate because some of them use special subsets, alternative candidate-generation methods, intermediate pipelines, or different evaluation settings.

The main categories are:

- `family_mentions/` — analyses of gene-family and ambiguous mentions
- `species_disambiguation/` — experiments on the role of species information
- `pubmedbert/` — PubMedBERT-based experiments
- `biosyn/` — BioSyn-based retrieval and linking experiments
- `synonym_rescue/` — dictionary and synonym-based recall recovery
- `hybrid_pipeline/` — combinations of multiple normalization components
- `selective_llm/` — experiments where the LLM is applied only to selected cases
- `other/` — exploratory scripts that do not fit the categories above

These experiments should not automatically be compared with the main results in `experiments/report/`.

Original historical scripts remain available under `legacy/`.
