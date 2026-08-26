import json
from pathlib import Path
from collections import defaultdict

IN = Path("family_pairwise_test_v4.jsonl")
OUT = Path("llm_rerank_prompts.jsonl")


groups = defaultdict(list)

with open(IN, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        key = (
            ex["pmid"],
            ex.get("start", ""),
            ex.get("end", ""),
            ex["mention"],
            ex["context"],
        )
        groups[key].append(ex)


with open(OUT, "w", encoding="utf-8") as out:
    for key, examples in groups.items():
        pmid, start, end, mention, context = key
        gold = examples[0]["gold_gene_ids"]

        candidates = []
        for ex in examples:
            candidates.append({
                "gene_id": ex["candidate_gene_id"],
                "tax_id": ex.get("candidate_tax_id", "UNKNOWN"),
                "symbol": ex.get("candidate_symbol", "UNKNOWN"),
                "description": ex.get("candidate_description", "UNKNOWN"),
                "aliases": ex.get("candidate_aliases", "UNKNOWN"),
            })

        prompt = """You are doing biomedical named entity normalization.

Task:
Given a gene/protein mention, its context, and a closed list of candidate NCBI Gene IDs, choose the most likely candidate.

Important rules:
- You must choose only one Gene ID from the candidate list.
- Do not invent new Gene IDs.
- Use the context, species clues, gene symbols, aliases, and descriptions.
- If the mention refers to a gene family like MAPK, ERK1/2, NF-kB, WNT, or HIF-1α, choose the candidate best supported by the local context.

Mention:
{mention}

Context:
{context}

Candidates:
{candidates}

Answer format:
GeneID: <one candidate gene_id>
Reason: <brief reason>
""".format(
            mention=mention,
            context=context,
            candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        )

        out.write(json.dumps({
            "pmid": pmid,
            "mention": mention,
            "context": context,
            "gold_gene_ids": gold,
            "candidates": candidates,
            "prompt": prompt,
        }, ensure_ascii=False) + "\n")

print("saved:", OUT)
print("prompts:", len(groups))
