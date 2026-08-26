# Gene symbol prediction and NCBI grounding

This experiment corresponds to Section 4.5.2 of the internship report.

## Goal

Direct generation of numerical GeneIDs performed poorly. This experiment therefore uses a gene symbol as an intermediate representation.

The pipeline is:

`mention + context -> Qwen -> gene symbol -> NCBI Gene lookup -> GeneID`

## Input

Qwen receives the same local context used in the direct GeneID experiment: up to 500 characters before and after the annotated mention.

## Model

Qwen2.5-14B-Instruct

The model predicts a gene symbol rather than a GeneID.

## Species information

The species information used during NCBI Gene lookup comes from the species-assignment component of GNormPlus.

It does not come from the NLM-Gene gold species annotations.

## NCBI Gene lookup

The predicted symbol is matched against a symbol and synonym table built from NCBI Gene.

The mapping procedure tries:

1. the predicted symbol with the species assigned by GNormPlus;
2. a global match when the symbol maps to only one GeneID;
3. a human GeneID when several mappings remain and a human entry is available;
4. the first remaining match as a final fallback.

If no matching symbol is found, the mention remains unmapped.

## Files

- `../../../../src/prompts/build_direct_prompts.py` prepares the symbol-prediction prompts.
- `../../../../src/data/build_ncbi_symbol_taxid_kb.py` builds the NCBI lookup table.
- `run_qwen.py` predicts gene symbols.
- `evaluate_grounding.py` maps the predicted symbols to GeneIDs and evaluates the final predictions.

## Result

Official test set: 2,729 mentions

Mapped to an NCBI GeneID: 2,641

Mapping coverage: **96.78%**

Correct GeneIDs: 1,521

Final accuracy: **55.73%**

The result refers to the complete symbol-prediction and grounding pipeline, not to symbol prediction alone.


## External species-assignment file

The historical species-aware grounding experiment uses document-level species assignments produced by GNormPlus.

For the cleaned repository, place this file locally at:

`data/external/gnormplus_tmp_SA.PubTator`

The file is treated as an external resource and is not committed to Git.
