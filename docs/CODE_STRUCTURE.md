# Code structure

The project was developed through many small experiments. The original working directory therefore contains several versions of similar scripts.

This repository keeps two views of the code.

## Clean project code

The main code is organized by purpose:

- `src/data/` prepares datasets and external resources.
- `src/retrieval/` generates candidate GeneIDs.
- `src/prompts/` builds model inputs.
- `src/inference/` contains reusable model inference code.
- `src/evaluation/` contains shared evaluation functions.
- `experiments/` describes how these components were combined in individual experiments.

File names describe the operation directly. For example:

- `build_contexts.py`
- `build_candidates.py`
- `run_qwen.py`
- `evaluate.py`

## Original scripts

`legacy/` contains copies of the scripts used while the experiments were being developed.

These files keep their original names and are not meant to represent the recommended project structure. They are included so that results can be traced back to the historical implementation.

Some original scripts contain fixed paths, experiment-specific filenames, or repeated code. These details are gradually replaced by command-line arguments in the cleaned version of the project.

