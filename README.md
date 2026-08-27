# Gene and Protein Normalization on NLM-Gene

This repository contains the experiments from an internship project on gene and protein entity normalization using the NLM-Gene benchmark.

The task starts from already annotated gene or protein mentions and assigns each mention to an NCBI GeneID. The project therefore focuses on entity normalization rather than named entity recognition.

The experiments compare rule-based normalization, direct language-model prediction, symbol grounding, dictionary and SapBERT candidate retrieval, and Qwen-based candidate selection.

## Main questions

The project investigates several questions:

- How well does GNormPlus perform under different evaluation protocols?
- Can a language model directly predict an NCBI GeneID?
- Is it more reliable to predict a gene symbol and ground it through NCBI Gene?
- How much candidate recall can be obtained from dictionary matching and SapBERT retrieval?
- How well can Qwen select the correct GeneID from a closed candidate list?
- How do textual context, candidate-set size, and candidate order affect selection?

## Dataset

The experiments use NLM-Gene.

The official training set contains 450 documents and the official test set contains 100 documents.

For development, the original training set was split at the document level:

| Split | Documents | Mentions |
|---|---:|---:|
| Train | 360 | 10,164 |
| Development | 90 | 2,518 |
| Official test | 100 | 2,729 |

The official test set was kept separate from development experiments.

## Main full-test results

The following results use the full official test set unless noted otherwise.

| Method | Final accuracy |
|---|---:|
| GNormPlus, report value | 54.56% |
| Direct Qwen GeneID generation | 10.33% |
| Qwen symbol prediction + species-aware NCBI grounding | 55.73% |
| Dictionary top-10 + Qwen | 42.51% |
| Dictionary top-20 + Qwen | 41.55% |
| SapBERT top-20 + Qwen | 23.56% |
| BioELQA-style T5-base noFT, top-5 | 4.51% |
| BioELQA-style Qwen noFT, top-5 | 13.89% |

For candidate-based methods, candidate recall and selector accuracy are also reported because final accuracy depends on both retrieval and selection.

The GNormPlus report value of 54.56% differs by one correct mention from the currently preserved GNormPlus output, which gives 54.53%. See `experiments/report/gnormplus_baseline/README.md` for details.

## Main pipeline

The candidate-based experiments follow the general structure:

```text
mention
   |
   v
candidate retrieval
(dictionary or SapBERT)
   |
   v
candidate list
   |
   v
prompt construction
   |
   v
Qwen candidate selection
   |
   v
selected GeneID
   |
   v
evaluation
```

## Repository structure

```text
.
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── prompts/
│   ├── inference/
│   └── evaluation/
│
├── experiments/
│   ├── report/
│   ├── post_report/
│   └── exploratory/
│
├── results/│
│   ├── post_report/
│   └── exploratory/
│
├── data/
│   ├── splits/
│   ├── examples/
│   ├── processed/
│   └── external/
│
├── docs/
├── legacy/
└── reports/
```

### `src/`

Contains cleaned and reusable implementations.

Examples:

- `src/retrieval/dictionary.py` — dictionary candidate generation
- `src/retrieval/sapbert.py` — SapBERT candidate retrieval
- `src/prompts/build_candidate_prompts.py` — candidate-selection prompt construction
- `src/inference/run_qwen_selector.py` — Qwen multiple-choice selection
- `src/evaluation/evaluate_candidates.py` — candidate recall, selector accuracy, and final accuracy

### `experiments/report/`

Contains the experiments described in the internship report.

Each directory explains the corresponding experiment and shows how the reusable scripts are combined.

### `experiments/post_report/`

Contains experiments run after the internship report was finalized.

These results are kept separate from the original report results.

### `experiments/exploratory/`

Contains additional investigations carried out during development.

### `legacy/`

Contains original experimental scripts preserved for traceability.

These files may contain repeated versions, fixed paths, intermediate experiments, and old output naming conventions. The cleaned implementations under `src/` should normally be used instead.

## Evaluation

For a dataset with \(N\) mentions:

```text
Final accuracy
= correct predictions / all mentions
```

For candidate-based methods:

```text
Recall@K
= mentions with a gold GeneID in the candidate list / all mentions
```

and:

```text
Selector accuracy
= correct predictions / mentions with a gold GeneID in the candidate list
```

Mentions with no candidates remain in the denominator of final accuracy and candidate recall.

## Experiment documentation

See:

- `docs/EXPERIMENTS.md` for the experiment sequence and results
- `docs/EXPERIMENT_MANIFEST.csv` for a machine-readable experiment index
- `docs/DATA_AND_RESOURCES.md` for datasets and external resources
- `docs/REPRODUCIBILITY.md` for historical implementation notes
- `docs/CODE_STRUCTURE.md` for the cleaned code organization
- `docs/PROJECT_OVERVIEW.md` for a longer project overview
- `docs/EXPLORATORY_INDEX.md` for selected exploratory experiments and their historical scripts

## Models and resources

The main external resources used in the project include:

- NLM-Gene
- NCBI Gene `gene_info`
- GNormPlus
- `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- T5-base
- Qwen2.5-Instruct models

Large datasets, model checkpoints, embeddings, and generated prediction files are not stored in this repository.

## Installation

Create a Python environment and install the main dependencies:

```bash
pip install -r requirements.txt
```

Individual experiment directories contain additional usage notes.

