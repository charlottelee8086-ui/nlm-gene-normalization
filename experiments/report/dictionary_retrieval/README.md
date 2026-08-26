# Dictionary candidate retrieval

This experiment corresponds to Section 4.6.1 of the internship report.

## Goal

The dictionary retriever generates a small set of possible NCBI GeneIDs for each annotated mention.

Unlike the SapBERT retriever, this method does not use semantic similarity. A candidate can only be retrieved when the normalized mention matches a symbol or synonym in the NCBI Gene dictionary.

## Dictionary

The lookup table contains:

- gene symbols and synonyms;
- NCBI GeneIDs;
- TaxIDs;
- alias frequencies.

Mention strings and dictionary terms are normalized by:

- converting text to lowercase;
- removing spaces;
- removing hyphens;
- removing underscores.

The normalized strings are then matched exactly.

## Candidate ranking

When several genes share the same alias, candidates are ordered using:

1. species priority: human, mouse, then rat;
2. alias frequency;
3. GeneID as a deterministic tie-breaker.

Other species are considered after these three priority species.

Duplicate GeneIDs are removed before the top-K candidates are returned.

## Candidate-set size

The same retrieval code can generate different candidate-set sizes.

The report includes experiments with:

- K = 10
- K = 15
- K = 20

## Usage

Example for top-10 candidates:

```bash
python src/retrieval/dictionary.py \
    --mentions bioelqa_dev_mentions.tsv \
    --dictionary ncbi_symbol_synonym_taxid_kb.tsv \
    --output dev_dictionary_top10.tsv \
    --k 10


```

For K = 15 or K = 20, only the `--k` argument needs to change.

## Evaluation

Candidate recall is computed over all mentions:

`mentions whose gold GeneID appears in the candidate list / all mentions`

Mentions with no retrieved candidates remain in the denominator.

