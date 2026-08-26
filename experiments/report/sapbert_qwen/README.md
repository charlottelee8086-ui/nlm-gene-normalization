# SapBERT candidates with Qwen selection

This experiment corresponds to Section 4.8.2 of the internship report.

## Goal

SapBERT first retrieves a list of possible GeneIDs. Qwen then selects one candidate using the mention and its textual context.

The pipeline is:

`mention -> SapBERT candidates -> Qwen selection -> GeneID`

## Candidate retrieval

Candidate lists are produced by `src/retrieval/sapbert.py`.

The historical development and test candidate files were produced with different SapBERT settings. These settings are described in `../sapbert_retrieval/README.md`.

## Prompt

For each mention, Qwen receives:

- the mention;
- its local context;
- the candidate GeneIDs;
- candidate species;
- the matched gene term.

The SapBERT experiment used slightly different prompt wording from the dictionary experiment. This wording is kept through the `sapbert` prompt template in `src/prompts/build_candidate_prompts.py`.

## Model

Qwen2.5-14B-Instruct

Inference is deterministic (`do_sample=False`).

## Results

Development set:

- Recall@20: 70.85%
- Selector accuracy: 65.58%
- Final accuracy: 46.47%

Official test set:

- Recall@20: 47.86%
- Selector accuracy: 49.23%
- Final accuracy: 23.56%

The development and test candidate lists came from different historical SapBERT configurations, so the difference between the two splits should not be interpreted as a pure generalization gap.

