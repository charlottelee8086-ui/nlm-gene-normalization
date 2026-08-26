# Reproducibility notes

This repository separates cleaned reusable code from the original experimental scripts.

The cleaned code is intended to make the main experiment logic easier to understand and rerun. The original files remain under `legacy/` when historical implementation details need to be checked.

## Dataset split

The original NLM-Gene training set was split at the document level.

- training: 360 documents, 10,164 mentions
- development: 90 documents, 2,518 mentions
- official test: 100 documents, 2,729 mentions

The official test set was kept separate from development experiments.

## Full-set denominators

For candidate-based methods:

```text
Recall@K = gold in candidates / all mentions
```

and:

```text
Final accuracy = correct / all mentions
```

Mentions with no generated candidate remain in the denominator.

Some historical intermediate scripts reported diagnostic values only over generated prompts. The cleaned evaluators use the complete mention set when reporting final accuracy and candidate recall.

## GNormPlus evaluation protocols

The project contains several GNormPlus evaluation views.

BELB-style linking accuracy is conditional on successful recognition of the gold mention span.

The currently preserved output gives:

```text
1488 / 1953 = 76.19%
```

The report's main full-test comparison instead uses all 2,729 gold mentions as the denominator.

The internship report records:

```text
54.56%
```

The currently preserved GNormPlus output contains 1,488 correct mention-level links:

```text
1488 / 2729 = 54.53%
```

This is a one-mention difference from the reported result. The exact historical source of the difference was not recovered.

A separate historical evaluator computes precision, recall, and F1 over `(document, start, end, GeneID)` tuples.

With the preserved output:

- precision: 73.88%
- recall: 44.29%
- F1: 55.38%

These tuple-level metrics are not the same as mention-level full-test accuracy.

## Species-aware symbol grounding

The species information used in the symbol-grounding experiment comes from the GNormPlus species-assignment output.

It is not a gold species annotation supplied directly by NLM-Gene.

The historical grounding order is:

1. exact symbol with the GNormPlus focus species;
2. unique GeneID globally;
3. human fallback;
4. first available candidate;
5. unmapped.

## SapBERT development and test configurations

The historical development and test candidate-retrieval scripts differ.

Development:

- NCBI `gene_info`
- mean pooling
- description included
- no GeneID deduplication after direct top-K retrieval

Test:

- aggregated synonym dictionary
- CLS pooling
- no description
- additional neighbors retrieved before GeneID deduplication

The development-to-test result difference should therefore not be interpreted as a controlled split comparison of one unchanged retriever.

## BioELQA-style Qwen model size

The internship report records Qwen2.5-14B-Instruct for the BioELQA-style no-finetuning experiment.

However, the saved prediction and evaluation filenames contain `qwen25_7b`.

The historical launch command is no longer available.

The model family is therefore verified, while the exact parameter size remains partially verified for this experiment.

Other custom Qwen candidate-selection scripts explicitly use Qwen2.5-14B-Instruct.

## Candidate-order experiment

The historical candidate-order experiment used seeds 1 through 5.

For every mention, Python's `random.Random(seed)` was used to permute candidate order.

The candidate metadata and prompt options received the same permutation, and option letters were reassigned.

Historical shuffled prompt, prediction, and mapping files were preserved.

A separate script used to compute all reported prediction-stability statistics was not located during cleanup. The cleaned analysis code therefore defines its denominators explicitly.

## Historical scripts

Files under `legacy/` may contain:

- absolute paths;
- repeated experiment versions;
- temporary debugging code;
- old output names;
- hard-coded development or test filenames;
- intermediate calculations later replaced by final report metrics.

They are preserved for traceability rather than as the recommended public interface of the repository.

## Python environment

The exact complete historical Python environment was not preserved as a lock file.

`requirements.txt` therefore lists the main runtime dependencies rather than claiming exact package-version reproduction.
