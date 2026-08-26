# Data and external resources

Large datasets and model files are not stored directly in this repository.

This document describes the main resources used by the experiments.

## NLM-Gene

The project uses the NLM-Gene benchmark for gene and protein normalization.

The official dataset contains:

- 450 training documents
- 100 test documents

The original training set was split at the document level for development:

| Split | Documents | Mentions |
|---|---:|---:|
| Train | 360 | 10,164 |
| Development | 90 | 2,518 |
| Official test | 100 | 2,729 |

The official test set was kept separate from development experiments.

The experiments start from annotated mention boundaries. The task addressed in this repository is normalization rather than named entity recognition.

## NCBI Gene

NCBI Gene is the target knowledge base.

The main external file used in the project is:

`gene_info`

It is used to construct:

- symbol lookup tables;
- synonym dictionaries;
- species-aware symbol grounding;
- SapBERT gene representations.

The original external file can be kept locally under:

`data/external/`

Generated dictionaries and processed tables can be stored under:

`data/processed/`

## GNormPlus

GNormPlus is used for:

- the rule-based baseline;
- document-level species assignment used by the historical symbol-grounding experiment.

GNormPlus itself is not redistributed in this repository.

Historical GNormPlus outputs must be provided separately when reproducing the baseline evaluations.

## SapBERT

The main model is:

`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`

The historical development and test retrieval configurations are not identical.

See:

`experiments/report/sapbert_retrieval/README.md`

before reproducing the reported retrieval values.

## Qwen

Most custom candidate-selection and direct-prediction experiments use:

`Qwen/Qwen2.5-14B-Instruct`

The exact Qwen model size in the BioELQA-style no-finetuning experiment remains historically uncertain.

See:

`experiments/report/bioelqa_no_finetuning/README.md`

## T5

The BioELQA-style no-finetuning comparison uses:

`t5-base`

## Generated files

The following files are normally regenerated and should not be committed when large:

- candidate tables;
- prompt files;
- model predictions;
- embeddings;
- model checkpoints;
- Hugging Face caches.

Small result summaries under `results/` are kept in the repository.

## Suggested local layout

```text
data/
├── splits/
├── examples/
├── processed/
└── external/
    └── gene_info
```
