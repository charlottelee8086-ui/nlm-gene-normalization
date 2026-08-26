# Candidate-set size

This experiment corresponds to Section 4.10.2 of the internship report.

## Goal

This experiment studies how the number of dictionary candidates affects the final selection result.

The textual context and Qwen settings are kept fixed. Only the number of retained dictionary candidates changes.

## Setup

Dataset: development set

Context: full abstract

Model: Qwen2.5-14B-Instruct

Candidate sizes:

- K = 10
- K = 15
- K = 20

The dictionary ranking is unchanged across the three settings. The top-10 candidates form a prefix of the top-15 list, and the top-15 candidates form a prefix of the top-20 list.

## Results

| K | Gold in candidates | Recall@K | Selector accuracy | Final accuracy |
|---:|---:|---:|---:|---:|
| 10 | 1,517 | 60.25% | 77.85% | 46.90% |
| 15 | 1,523 | 60.48% | 71.04% | 42.97% |
| 20 | 1,539 | 61.12% | 72.45% | 44.28% |

Increasing K improves candidate recall only slightly. The best final accuracy in this experiment is obtained with K = 10.

The final accuracy does not decrease monotonically: K = 20 performs better than K = 15, although both are below K = 10.

