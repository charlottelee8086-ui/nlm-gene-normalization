# Full-abstract evaluation on the official test set

It extends the dictionary-based candidate-selection pipeline by using the full abstract as context on the official NLM-Gene test set.

## Setup

Dataset: official NLM-Gene test set

Total mentions: 2,729

Candidate generation: NCBI Gene dictionary

Context: full abstract

Selector: Qwen2.5-14B-Instruct

Two candidate-set sizes were evaluated:

- K = 10
- K = 20

The retrieval method, prompt format, model, and decoding settings are otherwise the same as in the dictionary-based candidate-selection pipeline.

## Results

| K | Gold in candidates | Recall@K | Selector accuracy | Final accuracy |
|---:|---:|---:|---:|---:|
| 10 | 1,545 | 56.61% | 78.58% | 44.49% |
| 20 | 1,562 | 57.24% | 75.22% | 43.06% |

There are 582 mentions for which the dictionary retriever produces no candidate.

Compared with the earlier 500-character test setting, the full abstract improves the selector result for both candidate-set sizes.

## Code

The cleaned experiment reuses:

- `src/retrieval/dictionary.py`
- `src/prompts/build_candidate_prompts.py`
- `src/inference/run_qwen_selector.py`
- `src/evaluation/evaluate_candidates.py`

