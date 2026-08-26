#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

SPECIES_NAMES = {
    "9606": "human",
    "10090": "mouse",
    "10116": "rat",
    "7955": "zebrafish",
    "7227": "fruit fly",
    "3702": "arabidopsis",
    "6239": "worm",
    "4932": "yeast",
}

KEEP_TAXIDS = set(SPECIES_NAMES)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )


@torch.no_grad()
def encode_texts(
    texts,
    tokenizer,
    model,
    device,
    pooling,
    batch_size,
    max_length,
):
    embeddings = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="encoding",
    ):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)

        if pooling == "mean":
            emb = mean_pooling(
                outputs,
                inputs["attention_mask"],
            )
        elif pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        emb = torch.nn.functional.normalize(
            emb,
            p=2,
            dim=1,
        )

        embeddings.append(
            emb.cpu().numpy().astype("float32")
        )

    return np.vstack(embeddings)


def load_gene_info(path, max_terms=8):
    """Build the gene representations used in the report dev experiment."""
    rows = []

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="reading gene_info"):
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")

            if len(parts) < 9:
                continue

            tax_id = parts[0]
            gene_id = parts[1]
            symbol = parts[2]
            synonyms = parts[4]
            description = parts[8]

            if tax_id not in KEEP_TAXIDS:
                continue

            terms = []

            if symbol and symbol != "-":
                terms.append(symbol)

            if synonyms and synonyms != "-":
                for synonym in synonyms.split("|"):
                    synonym = synonym.strip()

                    if synonym and synonym != "-":
                        terms.append(synonym)

            clean_terms = []
            seen = set()

            for term in terms:
                key = term.lower()

                if key in seen:
                    continue

                seen.add(key)
                clean_terms.append(term)

                if len(clean_terms) >= max_terms:
                    break

            if not clean_terms:
                continue

            species = SPECIES_NAMES[tax_id]

            entity_text = " ".join(
                clean_terms + [species, description]
            )

            rows.append(
                {
                    "gene_id": str(gene_id),
                    "tax_id": str(tax_id),
                    "species": species,
                    "term": clean_terms[0],
                    "entity_text": entity_text,
                }
            )

    return pd.DataFrame(rows).drop_duplicates(
        ["gene_id", "tax_id"]
    )


def safe_int(value, default=0):
    try:
        return int(float(str(value)))
    except Exception:
        return default


def load_synonym_dictionary(path, max_terms=8):
    """Build the gene representations used in the report test experiment."""
    entity_terms = {}

    with open(path, encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n").split("\t")
        lower_header = [x.lower() for x in header]

        def find_column(names):
            for name in names:
                if name in lower_header:
                    return lower_header.index(name)
            return None

        term_index = find_column(
            ["term", "symbol", "name", "alias", "synonym"]
        )
        tax_index = find_column(["tax_id", "taxid"])
        gene_index = find_column(["gene_id", "geneid"])
        count_index = find_column(
            ["count", "freq", "frequency"]
        )

        if (
            term_index is None
            or tax_index is None
            or gene_index is None
        ):
            raise ValueError(
                "Could not identify the required dictionary columns."
            )

        for line in tqdm(f, desc="reading synonym dictionary"):
            parts = line.rstrip("\n").split("\t")

            if len(parts) <= max(
                term_index,
                tax_index,
                gene_index,
            ):
                continue

            term = parts[term_index].strip()
            tax_id = parts[tax_index].strip()
            gene_id = parts[gene_index].strip()

            if not term or not tax_id or not gene_id:
                continue

            if tax_id not in KEEP_TAXIDS:
                continue

            count = 0

            if (
                count_index is not None
                and count_index < len(parts)
            ):
                count = safe_int(parts[count_index])

            species = SPECIES_NAMES[tax_id]
            key = (gene_id, tax_id, species)

            if key not in entity_terms:
                entity_terms[key] = {}

            old_count = entity_terms[key].get(term)

            if old_count is None or count > old_count:
                entity_terms[key][term] = count

            # Keep memory use close to the historical implementation.
            if len(entity_terms[key]) > max_terms * 4:
                top_items = sorted(
                    entity_terms[key].items(),
                    key=lambda x: (-x[1], x[0]),
                )[:max_terms]

                entity_terms[key] = dict(top_items)

    rows = []

    for (gene_id, tax_id, species), term_counts in entity_terms.items():
        top_terms = [
            term
            for term, _ in sorted(
                term_counts.items(),
                key=lambda x: (-x[1], x[0]),
            )[:max_terms]
        ]

        if not top_terms:
            continue

        rows.append(
            {
                "gene_id": str(gene_id),
                "tax_id": str(tax_id),
                "species": species,
                "term": top_terms[0],
                "entity_text": " ".join(top_terms + [species]),
            }
        )

    return pd.DataFrame(rows)


def parse_gold(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def ensure_case_ids(df, prefix):
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [
            f"{prefix}_{i + 1}"
            for i in range(len(df))
        ]

    return df


def retrieve_report_dev(
    mentions,
    entities,
    mention_embeddings,
    entity_embeddings,
    k,
):
    """Historical dev retrieval: direct top-K without GeneID deduplication."""
    similarity = np.matmul(
        mention_embeddings,
        entity_embeddings.T,
    )

    rows = []

    for i, row in tqdm(
        mentions.iterrows(),
        total=len(mentions),
        desc="retrieving",
    ):
        top_indices = np.argsort(
            -similarity[i]
        )[:k]

        candidates = []

        for index in top_indices:
            item = entities.iloc[int(index)]

            candidates.append(
                "::".join(
                    [
                        str(item["gene_id"]),
                        str(item["tax_id"]),
                        str(item["species"]),
                        str(item["term"]),
                    ]
                )
            )

        output_row = row.to_dict()
        output_row["candidates"] = "|".join(candidates)
        rows.append(output_row)

    return pd.DataFrame(rows)


def retrieve_report_test(
    mentions,
    entities,
    mention_embeddings,
    entity_embeddings,
    k,
    batch_size,
):
    """Historical test retrieval: retrieve extra entries, then deduplicate GeneIDs."""
    rows = []

    retrieve_more = min(
        len(entities),
        k * 5,
    )

    for start in tqdm(
        range(0, len(mentions), batch_size),
        desc="retrieving",
    ):
        end = min(
            len(mentions),
            start + batch_size,
        )

        batch_mentions = mention_embeddings[start:end]

        scores = np.matmul(
            batch_mentions,
            entity_embeddings.T,
        )

        for local_index in range(end - start):
            row_index = start + local_index
            row = mentions.iloc[row_index]

            score_row = scores[local_index]

            if retrieve_more < len(score_row):
                top_indices = np.argpartition(
                    -score_row,
                    retrieve_more - 1,
                )[:retrieve_more]

                top_indices = top_indices[
                    np.argsort(-score_row[top_indices])
                ]
            else:
                top_indices = np.argsort(-score_row)

            candidates = []
            seen_gene_ids = set()

            for index in top_indices:
                item = entities.iloc[int(index)]
                gene_id = str(item["gene_id"])

                if gene_id in seen_gene_ids:
                    continue

                seen_gene_ids.add(gene_id)

                candidates.append(
                    "::".join(
                        [
                            gene_id,
                            str(item["tax_id"]),
                            str(item["species"]),
                            str(item["term"]),
                        ]
                    )
                )

                if len(candidates) >= k:
                    break

            output_row = row.to_dict()
            output_row["candidates"] = "|".join(candidates)
            rows.append(output_row)

    return pd.DataFrame(rows)


def print_recall(df, k):
    gold_column = None

    if "gold_geneid" in df.columns:
        gold_column = "gold_geneid"
    elif "gold_gene_ids" in df.columns:
        gold_column = "gold_gene_ids"

    if gold_column is None:
        return

    hits = 0

    for _, row in df.iterrows():
        gold = parse_gold(row[gold_column])

        candidate_geneids = {
            candidate.split("::", 1)[0]
            for candidate in str(row["candidates"]).split("|")
            if candidate
        }

        if gold & candidate_geneids:
            hits += 1

    total = len(df)

    print(
        f"Recall@{k}: "
        f"{hits}/{total} "
        f"({hits / total:.2%})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate SapBERT candidates for NLM-Gene mentions."
    )

    parser.add_argument(
        "--preset",
        required=True,
        choices=["report-dev", "report-test"],
        help="Historical SapBERT configuration to reproduce.",
    )
    parser.add_argument(
        "--mentions",
        required=True,
        help="Mention TSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output candidate TSV file.",
    )
    parser.add_argument(
        "--gene-info",
        default=None,
        help="NCBI gene_info file. Required for report-dev.",
    )
    parser.add_argument(
        "--dictionary",
        default=None,
        help="Symbol/synonym TaxID dictionary. Required for report-test.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    mentions = pd.read_csv(
        args.mentions,
        sep="\t",
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Preset:", args.preset)
    print("Device:", device)
    print("Model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    model = AutoModel.from_pretrained(
        args.model
    ).to(device)

    model.eval()

    if args.preset == "report-dev":
        if not args.gene_info:
            raise ValueError(
                "--gene-info is required for report-dev."
            )

        mentions = ensure_case_ids(
            mentions,
            "dev_sapbert_case",
        )

        entities = load_gene_info(
            args.gene_info
        )

        pooling = "mean"
        max_length = 96
        encode_batch_size = 128

    else:
        if not args.dictionary:
            raise ValueError(
                "--dictionary is required for report-test."
            )

        mentions = ensure_case_ids(
            mentions,
            "test_sapbert_case",
        )

        entities = load_synonym_dictionary(
            args.dictionary
        )

        pooling = "cls"
        max_length = 64
        encode_batch_size = 64

    print("Gene entries:", len(entities))
    print("Pooling:", pooling)

    entity_embeddings = encode_texts(
        entities["entity_text"].astype(str).tolist(),
        tokenizer,
        model,
        device,
        pooling,
        encode_batch_size,
        max_length,
    )

    mention_embeddings = encode_texts(
        mentions["mention"].astype(str).tolist(),
        tokenizer,
        model,
        device,
        pooling,
        encode_batch_size,
        max_length,
    )

    if args.preset == "report-dev":
        result = retrieve_report_dev(
            mentions,
            entities,
            mention_embeddings,
            entity_embeddings,
            args.k,
        )
    else:
        result = retrieve_report_test(
            mentions,
            entities,
            mention_embeddings,
            entity_embeddings,
            args.k,
            batch_size=128,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("Saved:", output_path)
    print("Total mentions:", len(result))

    print_recall(
        result,
        args.k,
    )


if __name__ == "__main__":
    main()
