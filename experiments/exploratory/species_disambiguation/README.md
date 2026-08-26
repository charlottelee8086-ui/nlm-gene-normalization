# Species disambiguation checks

These scripts were used to study the effect of species information during gene-symbol grounding.

## Symbol lookup without species

`evaluate_without_species.py` maps the predicted gene symbol directly through the NCBI symbol dictionary without using species information.

On the official test set:

- mapping coverage: 96.78%
- final GeneID accuracy: 2.16%

The mapping coverage is high because most predicted symbols can be found in the dictionary. The GeneID accuracy is much lower because the same gene symbol or synonym may refer to different genes across species.

## Species-aware lookup

The main symbol-grounding experiment uses species information produced by GNormPlus. Its final accuracy is 55.73%.

This comparison shows why finding a symbol in the dictionary is not enough. The species is often needed to select the correct NCBI Gene record.

