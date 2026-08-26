# Direct GeneID generation

This experiment corresponds to Section 4.5.1 of the internship report.

## Goal

The experiment tests whether Qwen can predict an NCBI GeneID directly from a gene or protein mention and its surrounding text.

## Input

Each example contains:

- the annotated gene or protein mention;
- up to 500 characters before the mention;
- up to 500 characters after the mention.

The mention boundaries come from NLM-Gene. This experiment therefore evaluates normalization rather than named entity recognition.

## Model

Qwen2.5-14B-Instruct

The model is asked to return a numerical NCBI GeneID.

## Files

- `../../../../src/prompts/build_direct_prompts.py` prepares the prompts and gold labels.
- `run_qwen.py` runs Qwen inference.
- `evaluate.py` compares the predicted GeneIDs with the NLM-Gene gold annotations.

## Evaluation

The official NLM-Gene test set contains 2,729 annotated mentions.

Final accuracy is computed over all test mentions:

`correct predictions / 2,729`

Missing or incorrect predictions are counted as errors.

## Result

Final accuracy: **10.33%**

The low result motivated the later symbol-grounding and candidate-selection experiments.

