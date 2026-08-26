from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

KB_PATH = Path("gnormplus_synonym_kb.tsv")
PRED_PATH = Path("pubmedbert_ner_test_predictions.tsv")
OUT_PATH = Path("biosyn_gnormplus_kb_linked.PubTator")

MODEL_NAME = "dmis-lab/biosyn-sapbert-bc2gn"

kb_mentions = []
kb_ids = []

with open(KB_PATH, encoding="utf-8") as f:
    for line in f:
        mention, gid = line.rstrip("\n").split("\t")
        kb_mentions.append(mention)
        kb_ids.append(gid)

print("KB size:", len(kb_mentions))

model = SentenceTransformer(MODEL_NAME)

kb_emb = model.encode(
    kb_mentions,
    batch_size=128,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

index = faiss.IndexFlatIP(kb_emb.shape[1])
index.add(kb_emb.astype("float32"))

pred_rows = []
pred_mentions = []

with open(PRED_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        pmid, start, end, mention, etype = parts[:5]
        if etype != "Gene":
            continue
        pred_rows.append((pmid, start, end, mention))
        pred_mentions.append(mention)

print("Pred mentions:", len(pred_mentions))

pred_emb = model.encode(
    pred_mentions,
    batch_size=128,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

scores, idxs = index.search(pred_emb.astype("float32"), 1)

with open(OUT_PATH, "w", encoding="utf-8") as out:
    for (pmid, start, end, mention), score, idx in zip(pred_rows, scores[:, 0], idxs[:, 0]):
        gid = kb_ids[idx]
        out.write(f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{gid}\t{score:.4f}\n")

print("Saved:", OUT_PATH)
