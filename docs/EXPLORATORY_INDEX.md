# Exploratory experiment index

The project contains a larger set of exploratory experiments in addition to the experiments selected for the internship report.

The complete historical scripts are preserved under `legacy/`. The directories under `experiments/exploratory/` provide a curated view of the most useful exploratory directions without duplicating every historical script version.

## Family mentions

Main historical scripts:

- `legacy/root/analyze_family_candidate_coverage.py`
- `legacy/root/analyze_family_gold_structure.py`
- `legacy/root/build_family_member_llm_prompts.py`
- `legacy/root/eval_family_member_llm_predictions.py`
- `legacy/root/export_family_hard_cases.py`

This work investigates difficult mentions that may refer to gene families, family members, or underspecified biological entities.

Several additional family reranker and prompt variants are preserved in `legacy/root/`.

## Species disambiguation

Cleaned code:

- `experiments/exploratory/species_disambiguation/evaluate_without_species.py`

Selected historical scripts:

- `legacy/root/analyze_oracle_wrong_species.py`
- `legacy/root/export_species_ambiguity_prompts.py`
- `legacy/root/eval_species_llm_predictions_strict.py`

These experiments study how species information affects GeneID grounding and normalization.

## PubMedBERT

Selected historical pipeline:

- `legacy/root/clean_pubmedbert_ner.py`
- `legacy/root/predict_pubmedbert_ner.py`
- `legacy/root/link_pubmedbert_predictions.py`
- `legacy/root/eval_pubmedbert_linked.py`

This branch explored PubMedBERT-based gene recognition and subsequent linking.

## BioSyn

Selected historical scripts:

- `legacy/root/dense_link_biosyn_gene.py`
- `legacy/root/eval_biosyn_gene.py`
- `legacy/root/dense_link_biosyn_gnormplus_filtered.py`
- `legacy/root/eval_biosyn_gnormplus_filtered.py`

These experiments explored BioSyn-style dense biomedical entity linking.

## Synonym rescue

Selected historical scripts:

- `legacy/root/build_rescue_dictionary.py`
- `legacy/root/analyze_single_gene_missing_rescue.py`
- `legacy/root/eval_gold_normalization_with_rescue.py`
- `legacy/root/eval_rescue_belb_style.py`

These experiments attempted to recover cases missed by the main normalization pipeline through additional symbol and synonym matching.

## Hybrid pipelines

Selected historical scripts:

- `legacy/root/build_hybrid_tmp_SA_cleaned.py`
- `legacy/root/eval_hybrid_cleaned.py`
- `legacy/root/eval_hybrid_pubmedbert_gnormplus.py`

These experiments combined outputs from multiple normalization components.

## Selective LLM use

Selected historical scripts:

- `legacy/root/analyze_llm_hurts.py`
- `legacy/root/simulate_selective_llm_gain.py`
- `legacy/root/compare_llm_vs_pubmedbert_trigger49.py`
- `legacy/root/eval_llm_trigger49_predictions.py`

These analyses investigate whether an LLM should be applied only to selected difficult cases rather than to every mention.

## Historical versions

The exploratory work evolved through many intermediate versions.

Files with names such as `v2`, `v3`, `v4`, `nocurrent`, and similar historical suffixes are intentionally left under `legacy/` instead of being promoted to the cleaned experiment directories.

This preserves the complete development history while keeping the public experiment structure readable.
