# SapBERT candidate retrieval

This experiment corresponds to Section 4.6.2 of the internship report.

## Goal

SapBERT was tested as a semantic alternative to exact dictionary matching.

Instead of requiring the mention to match an NCBI Gene symbol or synonym exactly, the mention and gene entries are encoded into the same embedding space. Candidate genes are then ranked by cosine similarity.

## Model

The experiments use:

`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`

Only eight common species are kept in the retrieval index:

- human
- mouse
- rat
- zebrafish
- fruit fly
- Arabidopsis
- worm
- yeast

## Historical development configuration

The development experiment builds one representation for each GeneID and TaxID from NCBI `gene_info`.

The gene representation contains:

- the gene symbol;
- up to eight symbols or synonyms;
- the species name;
- the gene description.

Mention and gene representations are encoded with SapBERT and pooled using mean pooling.

The development Recall@20 reported in the internship report is **70.85%**.

## Historical test configuration

The later test implementation uses the NCBI symbol/synonym lookup table rather than `gene_info`.

For each GeneID and TaxID, the representation contains:

- up to eight high-frequency symbols or synonyms;
- the species name.

The gene description is not included.

This version uses the SapBERT CLS representation rather than mean pooling.

The official test Recall@20 reported in the internship report is **47.86%**.

## Important implementation note

The development and test results were produced by related but not fully identical retrieval configurations.

For this reason, the repository keeps both historical settings explicit rather than treating the two results as a single unchanged retriever evaluated on different splits.

## Evaluation

Candidate recall is:

`mentions whose gold GeneID appears in the candidate list / all mentions`

All mentions remain in the denominator.


## Candidate deduplication

There is one additional difference between the historical development and test scripts.

The development script directly keeps the 20 highest-scoring GeneID-TaxID entries.

The test script first retrieves a larger set of nearby entries and then removes repeated GeneIDs before keeping the final top 20.

This difference is also preserved by the `report-dev` and `report-test` presets in `src/retrieval/sapbert.py`.
