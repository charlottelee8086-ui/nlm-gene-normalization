import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from collections import Counter
import re

train_path = Path("nlm_gene-train.arrow")
out_path = Path("train_recall_dictionary.txt")

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

def looks_ambiguous_family_term(m):
    m_low = m.lower()
    keywords = [
        "cytokine", "chemokine", "kinase", "mapk", "erk", "wnt",
        "tcr", "gpcr", "nf-kappa", "nf-κb", "interleukin",
        "receptor", "integrin", "complex", "family"
    ]
    return any(k in m_low for k in keywords)

counter = Counter()

for doc in read_arrow(train_path):
    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        if not ent.get("normalized"):
            continue

        mention = ent["text"][0].strip()

        # Avoid very short noisy terms unless they are known biomedical patterns
        if len(mention) <= 2:
            continue

        if looks_ambiguous_family_term(mention):
            counter[mention] += 1

with open(out_path, "w", encoding="utf-8") as f:
    for mention, count in counter.most_common():
        if count >= 2:
            f.write(mention + "\n")

print("Saved:", out_path)
print("Total terms:", sum(1 for _ in open(out_path, encoding="utf-8")))
print("\nTop terms:")
for m, c in counter.most_common(50):
    print(c, m)
