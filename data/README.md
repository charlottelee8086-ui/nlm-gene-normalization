# Data

The original datasets and large external resources are not stored in this repository.

## NLM-Gene

The experiments use the NLM-Gene benchmark.

The official dataset should be downloaded separately and converted to the format expected by the experiment scripts.

The project uses:

- 360 training documents with 10,164 mentions
- 90 development documents with 2,518 mentions
- 100 official test documents with 2,729 mentions

The train/development split was created at the document level from the original NLM-Gene training set.

## NCBI Gene

The project uses the NCBI Gene `gene_info` resource for:

- official gene symbols;
- gene synonyms and aliases;
- TaxIDs;
- species-aware symbol grounding;
- dictionary construction;
- SapBERT candidate representations.

Download `gene_info` from the official NCBI Gene resources and place it locally at:

`data/external/gene_info`

The file is not committed to Git.

Project-specific lookup tables can then be generated with scripts under:

`src/data/`

For example:

`src/data/build_ncbi_symbol_taxid_kb.py`

Generated tables should be stored under:

`data/processed/`

## GNormPlus species assignments

The historical symbol-grounding experiment also uses document-level species assignments produced by GNormPlus.

For the cleaned repository, place the corresponding PubTator file at:

`data/external/gnormplus_tmp_SA.PubTator`

This file is also treated as an external resource and is not committed to Git.

## Directory layout

```text
data/
├── README.md
├── external/
│   ├── gene_info
│   └── gnormplus_tmp_SA.PubTator
└── processed/
    └── generated project files
