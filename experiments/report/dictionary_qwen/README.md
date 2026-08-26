# Dictionary candidates with Qwen selection

This experiment corresponds to Section 4.8.1 of the internship report.

## Goal

The dictionary retriever usually returns several possible GeneIDs for an ambiguous mention. This experiment uses Qwen to choose one GeneID from that candidate list.

The pipeline is:

`mention -> dictionary candidates -> Qwen selection -> GeneID`

## Candidate generation

Candidates are produced by the dictionary retriever in `src/retrieval/dictionary.py`.

The report includes top-10 and top-20 test-set experiments.

## Prompt

For each mention, Qwen receives:

- the gene or protein mention;
- its textual context;
- the candidate GeneIDs;
- the species associated with each candidate;
- the matched dictionary term.

Candidates are presented as lettered options. Qwen must return one option and is not allowed to generate a GeneID outside the list.

## Model

Qwen2.5-14B-Instruct

Inference is deterministic (`do_sample=False`).

## Evaluation

Three measures are reported:

- **Candidate recall:** proportion of all mentions for which the gold GeneID is present in the candidate list.
- **Selector accuracy:** proportion of retrievable mentions for which Qwen selects the gold GeneID.
- **Final accuracy:** proportion of all mentions assigned the correct GeneID.

Mentions with no candidate remain in the denominator of candidate recall and final accuracy.

## Test results

| Candidates | Recall | Selector accuracy | Final accuracy |
|---|---:|---:|---:|
| Top 10 | 56.61% | 75.08% | 42.51% |
| Top 20 | 57.24% | 72.60% | 41.55% |

The larger candidate list slightly improved retrieval recall, but it also made selection harder.

