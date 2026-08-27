# Project overview

This repository contains the experiments carried out during an internship project on gene and protein normalization.

The main task is to assign an NCBI GeneID to a gene or protein mention that has already been identified in biomedical text. The experiments use the NLM-Gene dataset and focus on the normalization step rather than named entity recognition.

## Main questions

The project explored several questions:

- Can a large language model generate the correct GeneID directly?
- Is it easier to predict a gene symbol first and map it to NCBI Gene?
- How well can dictionary matching retrieve candidate genes?
- Can SapBERT improve candidate retrieval?
- How well can a language model select the correct GeneID from a candidate list?
- How much does textual context help candidate selection?
- How sensitive is the selector to candidate-set size and candidate order?

## Main experimental stages

### GNormPlus baseline

GNormPlus was first evaluated as a rule-based reference system. Two evaluation settings were used: a BELB-style filtered evaluation and a full-test evaluation over all NLM-Gene test mentions.

### Direct GeneID generation

Qwen2.5-14B-Instruct received a gene or protein mention together with its surrounding context and generated a GeneID directly.

### Gene symbol grounding

Instead of generating a numerical identifier, Qwen predicted a gene symbol. The symbol was then mapped to NCBI Gene using species information produced by the species-assignment component of GNormPlus.

### Dictionary retrieval

Gene candidates were generated from NCBI Gene symbols and synonyms. Mention strings were normalized before exact lookup. Candidates were ranked using species priority and alias frequency.

### SapBERT retrieval

SapBERT was tested as a semantic alternative to dictionary matching. Gene entries were represented using their names, synonyms, and species information before nearest-neighbor retrieval.

The historical development and test implementations were not fully identical. The development experiment used mean pooling and included gene descriptions, while the later test implementation used the CLS representation and omitted descriptions. Both versions are kept in the repository for reproducibility.

### BioELQA-style no-finetuning experiment

A compact multiple-choice setting inspired by BioELQA was tested without task-specific fine-tuning. T5-base and Qwen scored answer options from fixed SapBERT top-5 candidate lists.

This experiment should not be interpreted as a reproduction of the full BioELQA training procedure.

### Qwen candidate selection

Qwen was also used with richer multiple-choice prompts. These prompts included the mention, textual context, and information about each candidate.

Dictionary and SapBERT candidate lists were evaluated separately.

### Context ablation

The same SapBERT top-5 candidates were kept fixed while the amount of textual context shown to Qwen was changed. The tested settings included the mention alone, sentence-level context, a three-sentence window, a 500-character window, the full abstract, and the title plus abstract.

### Candidate-set size

Dictionary candidate lists of different sizes were tested to study the trade-off between candidate recall and selection difficulty.

### Candidate-order sensitivity

The order of the same top-10 dictionary candidates was randomly changed five times. This experiment checked whether Qwen relied strongly on fixed answer positions.

## Follow-up experiments

After the internship report was completed, the dictionary top-10 and top-20 pipelines were also evaluated on the official test set using the full abstract as context.

These follow-up results are stored separately from the results reported in the internship report.

## Repository organization

- `src/` contains reusable implementations.
- `experiments/report/` contains experiments described in the internship report.
- `experiments/additional_experiments/` contains experiments run after the report was completed.
- `experiments/exploratory/` contains additional experiments that were investigated during the project.
- `legacy/` keeps the original scripts used during development.
- `results/` contains compact result tables.
- `data/` contains data preparation instructions and small metadata files.
- `reports/` contains the final written reports.
