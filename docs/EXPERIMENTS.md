# Experiments

This document summarizes the main experiments in the project.

The experiments are divided into three groups:

1. experiments included in the internship report;
2. experiments run after the report;
3. exploratory experiments carried out during development.

## Report experiments

### GNormPlus baseline

Location:

`experiments/report/gnormplus_baseline/`

The BELB-style evaluation and the full-test evaluation use different denominators.

For the currently preserved GNormPlus output:

- BELB-style linking accuracy: 76.19%
- mention-level full-test accuracy: 54.53%

The internship report records 54.56% for the full-test result. This differs from the currently preserved output by one correct mention and is documented in the experiment README.

### Direct GeneID generation

Location:

`experiments/report/direct_geneid/`

Model:

`Qwen2.5-14B-Instruct`

Official test result:

- final accuracy: 10.33%

The model directly predicts the numerical NCBI GeneID.

### Gene-symbol prediction and NCBI grounding

Location:

`experiments/report/symbol_grounding/`

Pipeline:

```text
mention + context
→ Qwen gene-symbol prediction
→ GNormPlus species assignment
→ NCBI Gene lookup
→ GeneID
```

Official test results:

- mapped: 2,641 / 2,729
- mapping coverage: 96.78%
- correct: 1,521
- final accuracy: 55.73%

The species information comes from GNormPlus species-assignment output rather than a gold NLM-Gene species field.

### Dictionary retrieval

Location:

`experiments/report/dictionary_retrieval/`

Dictionary candidate generation uses normalized NCBI Gene symbols and synonyms.

### SapBERT retrieval

Location:

`experiments/report/sapbert_retrieval/`

Model:

`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`

Historical development and test retrieval configurations differ.

Development:

- source: NCBI `gene_info`
- mean pooling
- description included
- no GeneID deduplication after top-K retrieval

Test:

- source: aggregated synonym dictionary
- CLS pooling
- no description
- GeneID deduplication after retrieval

Reported Recall@20:

- development: 70.85%
- test: 47.86%

Because the historical retrieval implementations differ, this development-to-test difference should not be treated as a controlled comparison of one unchanged retriever.

### BioELQA-style no-finetuning

Location:

`experiments/report/bioelqa_no_finetuning/`

The experiment uses a compact multiple-choice input without surrounding context.

Results:

| Model | Split | Recall@5 | Selector accuracy | Final accuracy |
|---|---|---:|---:|---:|
| T5-base | Development | 27.12% | 14.64% | 3.97% |
| T5-base | Test | 29.68% | 15.19% | 4.51% |
| Qwen2.5-Instruct | Development | 27.12% | 35.72% | 9.69% |
| Qwen2.5-Instruct | Test | 29.68% | 46.79% | 13.89% |

The exact Qwen model size is historically uncertain: the report states 14B, while saved result filenames contain `qwen25_7b`.

### Dictionary candidates with Qwen

Location:

`experiments/report/dictionary_qwen/`

Official test results:

| K | Recall@K | Selector accuracy | Final accuracy |
|---:|---:|---:|---:|
| 10 | 56.61% | 75.08% | 42.51% |
| 20 | 57.24% | 72.60% | 41.55% |

### SapBERT candidates with Qwen

Location:

`experiments/report/sapbert_qwen/`

Results:

| Split | Recall@20 | Selector accuracy | Final accuracy |
|---|---:|---:|---:|
| Development | 70.85% | 65.58% | 46.47% |
| Test | 47.86% | 49.23% | 23.56% |

### Context ablation

Location:

`experiments/report/context_ablation/`

The same SapBERT top-5 candidate list is kept fixed across all context settings.

Recall@5 is 45.15%.

| Context | Selector accuracy | Final accuracy |
|---|---:|---:|
| Mention only | 62.09% | 28.04% |
| Sentence | 69.66% | 31.45% |
| Three sentences | 70.54% | 31.85% |
| 500-character window | 72.56% | 32.76% |
| Full abstract | 75.99% | 34.31% |
| Title + abstract | 75.46% | 34.07% |

### Full-abstract dictionary context

Location:

`experiments/report/dictionary_full_abstract/`

| Context | Recall@10 | Selector accuracy | Final accuracy |
|---|---:|---:|---:|
| 500-character window | 60.25% | 74.88% | 45.12% |
| Full abstract | 60.25% | 77.85% | 46.90% |

### Candidate-set size

Location:

`experiments/report/candidate_size/`

| K | Gold in candidates | Recall@K | Selector accuracy | Final accuracy |
|---:|---:|---:|---:|---:|
| 10 | 1,517 | 60.25% | 77.85% | 46.90% |
| 15 | 1,523 | 60.48% | 71.04% | 42.97% |
| 20 | 1,539 | 61.12% | 72.45% | 44.28% |

Increasing K improves candidate recall slightly, but the best final accuracy is obtained with K = 10.

### Candidate-order sensitivity

Location:

`experiments/report/candidate_order/`

Five random candidate permutations were evaluated.

Shuffle mean:

- selector accuracy: 78.38 ± 0.55%
- final accuracy: 47.22 ± 0.33%
- same GeneID as the original prediction: 71.21 ± 0.55%

When the originally selected GeneID moved to another option position:

- followed the same GeneID: 70.04 ± 0.70%
- repeated the old option letter: 4.75 ± 0.79%

The overall result is relatively stable, although individual predictions can change with candidate order.

## Post-report experiments

### Full-abstract official test

Location:

`experiments/post_report/full_abstract_test/`

These experiments were run after the internship report was completed.

| K | Recall@K | Selector accuracy | Final accuracy |
|---:|---:|---:|---:|
| 10 | 56.61% | 78.58% | 44.49% |
| 20 | 57.24% | 75.22% | 43.06% |

These values are kept separate from the results originally reported in the internship report.

## Exploratory experiments

Additional experiments are kept under:

`experiments/exploratory/`

They include work on:

- family mentions
- species disambiguation
- PubMedBERT
- BioSyn
- synonym rescue
- hybrid pipelines
- selective LLM use

These experiments are preserved as part of the project history but should not automatically be compared with the main report results because their subsets and evaluation settings may differ.
