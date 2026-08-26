# BioELQA-style no-finetuning experiment

This experiment corresponds to Section 4.7 of the internship report.

## Goal

This experiment tests a compact multiple-choice formulation inspired by BioELQA.

It is not a full reproduction of BioELQA. The original training procedure was not reproduced, and neither language model was fine-tuned for this task.

## Input

Each example contains:

- the gene or protein mention;
- five SapBERT candidate options;
- an answer field.

No surrounding textual context is included.

Each option corresponds to an NCBI GeneID, but the model selects an answer letter rather than generating the identifier directly.

## Candidate lists

SapBERT provides the top-5 candidate list.

Within each dataset split, T5-base and Qwen use exactly the same candidate lists. This keeps the retrieval stage fixed when their selection results are compared.

The candidate lists used in this experiment were generated with a different SapBERT configuration from the candidate lists used later in the context ablation. The two experiments should therefore not be treated as a controlled comparison of context alone.

## Answer-letter scoring

For each example, the possible answers are:

`A, B, C, D, E`

The selected answer is the letter with the highest model likelihood.

T5-base uses sequence-to-sequence likelihood scoring.

Qwen treats the prompt as an input prefix and scores each possible answer letter as a continuation.

No free-form GeneID generation is used.

## Models

- T5-base
- Qwen2.5-Instruct

The internship report records Qwen2.5-14B-Instruct for this experiment. However, the saved Qwen result filenames contain `qwen25_7b`, and the original launch command is no longer available. The repository therefore keeps the Qwen model size marked as historically uncertain rather than assigning it without evidence.

## Results

| Model | Split | Recall@5 | Selector accuracy | Final accuracy |
|---|---|---:|---:|---:|
| T5-base | Development | 27.12% | 14.64% | 3.97% |
| T5-base | Test | 29.68% | 15.19% | 4.51% |
| Qwen2.5-Instruct | Development | 27.12% | 35.72% | 9.69% |
| Qwen2.5-Instruct | Test | 29.68% | 46.79% | 13.89% |

The low final accuracy is partly explained by candidate recall. When the gold GeneID is absent from the top-5 list, neither selector can recover it.


## Cleaned implementation

The cleaned prompt builder keeps one gold row for every input mention, including mentions with no candidate list.

This makes candidate recall and final accuracy use the full dataset as their denominator.

The original development scripts are still available under `legacy/` for comparison with the historical runs.
