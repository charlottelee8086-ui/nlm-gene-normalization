# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEV = Path("../bioelqa_dev_mentions.tsv")
KB = Path("../ncbi_symbol_synonym_taxid_kb.tsv")

OUT = Path("bioelqa_dev_candidates_sapbert_genelevel_top20.tsv")

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

TOPK = 20
BATCH_SIZE = 128
MAX_SYNONYMS_PER_GENE = 8

SPECIES_NAME = {
    "9606": "human",
    "10090": "mouse",
    "10116": "rat",
    "7955": "zebrafish",
    "7227": "fruit fly",
    "3702": "arabidopsis",
    "6239": "worm",
    "4932": "yeast",
}

PRIOR_TAX = {
    "9606": 0,    # human
    "10090": 1,   # mouse
    "10116": 2,   # rat
    "7955": 3,    # zebrafish
    "7227": 4,    # fly
    "3702": 5,    # arabidopsis
    "6239": 6,    # worm
    "4932": 7,    # yeast
}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)


def load_genelevel_entities(kb_path):
    """
    Input KB format expected:
    term    tax_id    gene_id    count

    We group by (gene_id, tax_id). Each GeneID/species pair becomes one entity.
    Entity text = top terms + species.
    """
    df = pd.read_csv(kb_path, sep="\t")

    print("KB columns:", list(df.columns))
    print("Raw KB rows:", len(df))

    df["term"] = df["term"].astype(str)
    df["tax_id"] = df["tax_id"].astype(str)
    df["gene_id"] = df["gene_id"].astype(str)

    if "count" not in df.columns:
        df["count"] = 1

    try:
        df["count"] = df["count"].astype(int)
    except Exception:
        df["count"] = 1

    entities = []

    grouped = df.groupby(["gene_id", "tax_id"])

    for (gene_id, tax_id), g in tqdm(grouped, desc="grouping gene entities"):
        g = g.sort_values("count", ascending=False)

        terms = []
        seen = set()
        for t in g["term"].tolist():
            t = str(t).strip()
            key = t.lower()
            if not t or key in seen:
                continue
            seen.add(key)
            terms.append(t)
            if len(terms) >= MAX_SYNONYMS_PER_GENE:
                break

        if not terms:
            continue

        species = SPECIES_NAME.get(tax_id, "taxid_" + tax_id)

        # BioELQA-like entity text: one candidate entity represented by names/synonyms.
        entity_text = " ".join(terms + [species])

        entities.append({
            "gene_id": gene_id,
            "tax_id": tax_id,
            "species": species,
            "term": terms[0],
            "entity_text": entity_text,
        })

    ent_df = pd.DataFrame(entities)

    # Optional priority sorting: human/mouse/rat first if similarity ties.
    ent_df["tax_priority"] = ent_df["tax_id"].map(lambda x: PRIOR_TAX.get(str(x), 99))

    print("Gene-level entities:", len(ent_df))
    print(ent_df.head())

    return ent_df


def parse_gold(x):
    return set(str(x).split("|"))


def main():
    print("Loading gene-level KB entities...")
    ent_df = load_genelevel_entities(KB)

    print("Loading dev mentions...")
    dev = pd.read_csv(DEV, sep="\t")
    print("Dev rows:", len(dev))
    print("Dev columns:", list(dev.columns))

    gold_col = "gold_geneid" if "gold_geneid" in dev.columns else "gold_gene_ids"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading SapBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("Encoding gene-level entities...")
    entity_texts = ent_df["entity_text"].astype(str).tolist()
    entity_embs = encode_texts(entity_texts, tokenizer, model, device)

    print("Encoding dev mentions...")
    mention_texts = dev["mention"].astype(str).tolist()
    mention_embs = encode_texts(mention_texts, tokenizer, model, device)

    print("Retrieving candidates...")
    sim = np.matmul(mention_embs, entity_embs.T)

    rows = []
    oracle_hit = 0
    total = 0

    for i, r in tqdm(dev.iterrows(), total=len(dev), desc="writing"):
        scores = sim[i]

        # retrieve more than TOPK so we can apply tie/priority handling if needed
        top_idx = np.argsort(-scores)[:TOPK]

        candidates = []
        for j in top_idx:
            item = ent_df.iloc[j]
            candidates.append(
                f'{item["gene_id"]}::{item["tax_id"]}::{item["species"]}::{item["term"]}'
            )

        gold = parse_gold(r[gold_col])
        cand_gids = set(c.split("::")[0] for c in candidates)

        total += 1
        if gold & cand_gids:
            oracle_hit += 1

        out = r.to_dict()
        out["candidates"] = "|".join(candidates)
        rows.append(out)

    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)

    print("Saved:", OUT)
    print("Total:", total)
    print("Oracle hit:", oracle_hit)
    print("Candidate recall:", oracle_hit / total if total else 0)


if __name__ == "__main__":
    main()
