# Dictionary selection with full-abstract context

This experiment corresponds to Section 4.10.1 of the internship report.

## Goal

The initial dictionary candidate-selection experiment used a 500-character window around the mention.

This follow-up experiment keeps the same dictionary candidates, candidate order, prompt instructions, Qwen model, and decoding settings, but replaces the local window with the full abstract.

The purpose is to measure the effect of giving the selector broader context without changing candidate retrieval.

## Model

Qwen2.5-14B-Instruct

## Candidates

The experiment uses the same dictionary top-10 ranking as the earlier development experiment.

Because candidate generation is unchanged, Recall@10 remains **60.25%**.

## Results

| Context | Recall@10 | Selector accuracy | Final accuracy |
|---|---:|---:|---:|
| 500-character window | 60.25% | 74.88% | 45.12% |
| Full abstract | 60.25% | 77.85% | 46.90% |

The full abstract improves the selection stage while leaving retrieval unchanged.

## Code

The experiment reuses:

- `src/retrieval/dictionary.py`
- `src/prompts/build_candidate_prompts.py`
- `src/inference/run_qwen_selector.py`
- `src/evaluation/evaluate_candidates.py`

