# Context ablation

This experiment corresponds to Section 4.9 of the internship report.

## Goal

The experiment studies how much textual context helps Qwen choose between candidate GeneIDs.

The candidate list is kept fixed for every mention. Only the text shown to the model changes.

This makes the experiment a comparison of context settings rather than a comparison of retrieval methods.

## Dataset

The experiment is run on the development set only.

Total mentions: 2,518

The gold GeneID appears in the fixed SapBERT top-5 list for 1,137 mentions.

Recall@5 is therefore **45.15%** for every context setting.

## Context settings

The following inputs were compared:

- mention only;
- sentence containing the mention;
- three-sentence window;
- 500-character window;
- full abstract;
- title and abstract.

## Results

| Context | Correct | Final accuracy | Selector accuracy | Invalid outputs |
|---|---:|---:|---:|---:|
| Mention only | 706 | 28.04% | 62.09% | 192 |
| Sentence | 792 | 31.45% | 69.66% | 380 |
| Three sentences | 802 | 31.85% | 70.54% | 450 |
| 500-character window | 827 | 32.76% | 72.56% | 458 |
| Full abstract | 864 | 34.31% | 75.99% | 500 |
| Title + abstract | 858 | 34.07% | 75.46% | 500 |

The full abstract gives the highest final accuracy in this experiment.

Adding the title does not improve the result further.

## Interpretation

More context generally helps the selector distinguish candidates, especially when the mention alone does not reveal the relevant species or gene.

Context cannot correct a retrieval failure. If the gold GeneID is missing from the fixed top-5 candidate list, the selector cannot choose it.


## Files

- `src/data/attach_contexts.py` attaches all context variants to the fixed candidate file.
- `src/prompts/build_context_prompts.py` builds one prompt set for a chosen context setting.
- `src/inference/run_qwen_context.py` runs Qwen2.5-14B-Instruct with the historical decoding settings.
- `src/evaluation/evaluate_letter_selection.py` maps answer letters back to GeneIDs and computes the evaluation metrics.
- `results/report/context_ablation.csv` contains the results reported in the internship report.

## Reproduction

First attach the context variants and build the six prompt sets:

```bash
experiments/report/context_ablation/prepare.sh \
    <fixed_candidate_file.tsv> \
    <development_context_file.tsv>
```
