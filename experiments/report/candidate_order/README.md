# Candidate-order sensitivity

This experiment corresponds to Section 4.10.3 of the internship report.

## Goal

The dictionary ranking often places strong candidates near the beginning of the list.

This experiment checks whether Qwen depends strongly on those original option positions.

## Setup

Dataset: development set

Candidates: dictionary top-10

Context: full abstract

Model: Qwen2.5-14B-Instruct

The candidate set for each mention is kept unchanged. Only the order of the candidates is changed.

Five shuffled versions of the candidate lists were evaluated.

After shuffling, the answer letters are reassigned according to the new candidate positions.

## Results

| Setting | Selector accuracy | Final accuracy | Same GeneID | Same letter |
|---|---:|---:|---:|---:|
| Original order | 77.85% | 46.90% | - | - |
| Shuffle 1 | 78.64% | 47.38% | 72.00% | 14.49% |
| Shuffle 2 | 77.72% | 46.82% | 70.83% | 14.98% |
| Shuffle 3 | 78.11% | 47.06% | 70.61% | 16.77% |
| Shuffle 4 | 78.25% | 47.14% | 71.16% | 15.74% |
| Shuffle 5 | 79.17% | 47.70% | 71.44% | 14.96% |
| Mean ± std. | 78.38 ± 0.55% | 47.22 ± 0.33% | 71.21 ± 0.55% | 15.39 ± 0.89% |

The overall accuracy stays close to the original result after shuffling.

Individual predictions are less stable. Across the five runs, Qwen selects the same GeneID as in the original ordering in about 71% of cases.

When the originally selected GeneID moves to another position, Qwen follows that GeneID to its new position in 70.04 ± 0.70% of cases. It repeats the old answer letter in only 4.75 ± 0.79% of those cases.

These results do not show a strong fixed-letter preference, although candidate order can still change individual predictions.


## Implementation

The historical experiment used Python's `random.Random(seed)` with seeds 1 through 5.

For each mention, the candidate list was randomly permuted. The same permutation was applied to the candidate metadata and the prompt options, after which the answer letters were reassigned.

The original script also stored a map from the old option position to the new option position for each GeneID.

The cleaned implementation performs the shuffle on the candidate TSV first and then rebuilds the prompt. This avoids editing prompt text with regular expressions while preserving the same experimental operation.

## Consistency analysis

The historical result files contain the five shuffled candidate maps and prediction files, but a separate script used to compute the reported prediction-stability statistics was not found during repository cleanup.

For this reason, `src/evaluation/analyze_candidate_order.py` states its denominators explicitly.

- `Same GeneID` compares cases where both the original and shuffled runs produced a valid option.
- `Same letter` uses the same set of comparable cases.
- The moved-candidate analysis only includes cases where the GeneID selected in the original run moved to a different option position after shuffling.

The result tables reported in the internship report are kept unchanged under `results/report/`.
