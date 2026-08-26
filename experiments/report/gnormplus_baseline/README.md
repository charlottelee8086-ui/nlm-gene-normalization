# GNormPlus baseline

This experiment corresponds to Section 4.3 of the internship report.

GNormPlus was used as a reference system before the LLM-based normalization experiments.

Several evaluation views were used during the project. Their denominators are different, so they should not be treated as interchangeable scores.

## BELB-style linking evaluation

The BELB-style evaluation first checks whether GNormPlus produced a gene prediction at the exact gold mention span.

Linking accuracy is then computed only over recognized gold spans:

`correct links / recognized gold spans`

For the preserved GNormPlus output:

- gold mentions: 2,729
- recognized gold spans: 1,953
- correct links: 1,488
- wrong links: 465
- linking accuracy: **76.19%**

The published BELB reference result is **76.00%**.

This reproduced result therefore closely matches the published BELB-style reference value.

## Mention-level full-test evaluation

For comparison with the other normalization methods, every annotated NLM-Gene test mention is kept in the denominator:

`correct links / all gold mentions`

Missing spans and incorrect GeneIDs are counted as errors.

With the currently preserved GNormPlus output:

- total gold mentions: 2,729
- correct links: 1,488
- full-test accuracy: **54.53%**

The internship report records **54.56%** for this experiment.

A value of 54.56% over 2,729 mentions corresponds to 1,489 correct predictions, whereas the currently preserved GNormPlus output gives 1,488. This is a one-mention historical discrepancy. The exact source of that difference could not be recovered, so the repository keeps both the reported value and the currently reproducible value rather than silently replacing either one.

## Tuple-level precision, recall, and F1

The historical `eval_gnorm2.py` script used another evaluation view. It represents each acceptable annotation as a separate `(document, start, end, GeneID)` tuple.

With the currently preserved output:

- gold tuples: 3,360
- predicted tuples: 2,014
- true positives: 1,488
- false positives: 526
- false negatives: 1,872
- precision: 73.88%
- recall: 44.29%
- F1: 55.38%

These tuple-level metrics are useful for reproducing the historical script, but the F1 score should not be confused with the mention-level full-test accuracy used in the main result comparison.

## Species focus used in the historical setup

The helper script used to prepare the GNormPlus focus input copied document-level species assignments from an existing GNormPlus `tmp_SA` file.

When no species assignment was found for a document, the historical script used human (TaxID 9606) as the default.

This behavior is preserved in `prepare_focus_input.py`.

## Code

The original scripts are preserved under `legacy/`.

The cleaned evaluation code is in:

- `src/evaluation/evaluate_gnormplus.py`

The three evaluation views can be run separately with:

- `evaluate_belb_style.sh`
- `evaluate_full_test.sh`
- `evaluate_tuple_prf.sh`

