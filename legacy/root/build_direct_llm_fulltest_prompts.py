# -*- coding: utf-8 -*-

import json
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

GOLD_ARROW = Path("nlm_gene-test.arrow")
OUT_JSONL = Path("direct_llm_fulltest_prompts.jsonl")
OUT_GOLD = Path("direct_llm_fulltest_gold.tsv")


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def get_doc_text(doc):
    texts = []
    for p in doc["passages"]:
        texts.append(p["text"][0])
    return " ".join(texts)


def window_context(text, start, end, window=500):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


rows = read_arrow(GOLD_ARROW)

case_id = 0

with open(OUT_JSONL, "w", encoding="utf-8") as out, \
     open(OUT_GOLD, "w", encoding="utf-8") as gold_out:

    gold_out.write("case_id\tpmid\tmention\tgold_gene_ids\n")

    for doc in rows:
        pmid = str(doc["document_id"])
        full_text = get_doc_text(doc)

        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            if not ent.get("normalized"):
                continue

            gids = []
            for norm in ent["normalized"]:
                if norm.get("db_name") == "NCBIGene":
                    gids.append(str(norm["db_id"]))

            if not gids:
                continue

            mention = ent["text"][0].strip()
            start, end = ent["offsets"][0]
            context = window_context(full_text, int(start), int(end))

            case_id += 1
            cid = f"direct_case_{case_id}"

            prompt_geneid = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention and its local context, predict the most likely NCBI Gene ID.

Important:
- Output one NCBI Gene ID if possible.
- Do not explain.
- If uncertain, still provide the most likely Gene ID.

Mention:
{mention}

Context:
{context}

Answer format:
{cid}    GeneID: <gene_id>
"""

            prompt_symbol = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention and its local context, predict the most likely official gene symbol.

Important:
- Output one gene symbol if possible.
- Do not output a Gene ID.
- Do not explain.
- Use the context to resolve ambiguity.

Mention:
{mention}

Context:
{context}

Answer format:
{cid}    Symbol: <gene_symbol>
"""

            out.write(json.dumps({
                "case_id": cid,
                "pmid": pmid,
                "mention": mention,
                "context": context,
                "prompt_geneid": prompt_geneid,
                "prompt_symbol": prompt_symbol,
            }, ensure_ascii=False) + "\n")

            gold_out.write(
                f"{cid}\t{pmid}\t{mention}\t{'|'.join(sorted(set(gids)))}\n"
            )

print("saved:", OUT_JSONL)
print("saved:", OUT_GOLD)
print("cases:", case_id)
