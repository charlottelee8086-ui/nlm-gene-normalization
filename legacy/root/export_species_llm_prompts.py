import json

IN = "species_ambiguity_with_context.jsonl"
PROMPTS = "species_llm_prompts.txt"
GOLD = "species_llm_gold.tsv"

with open(IN, encoding="utf-8") as f, \
     open(PROMPTS, "w", encoding="utf-8") as out_p, \
     open(GOLD, "w", encoding="utf-8") as out_g:

    out_g.write("case_id\tpmid\tmention\tgold_gene_ids\tfocus_taxid\tpred_taxid\n")

    for line in f:
        ex = json.loads(line)

        prompt = """CASE_ID: {case_id}

You are helping with biomedical gene normalization.

Task:
Identify the most likely species or organism context for this gene/protein mention.

Important:
- Do NOT choose a Gene ID.
- Only infer the species from the context.
- Use explicit context clues such as mice, rats, human cells, patients, Arabidopsis, C. elegans, etc.
- If the context is ambiguous, answer unclear.

Options:
- human / 9606
- mouse / 10090
- rat / 10116
- Arabidopsis / 3702
- worm / 6239
- other
- unclear

Mention:
{mention}

Context:
{context}

Answer format:
{case_id}    Species: <human|mouse|rat|Arabidopsis|worm|other|unclear>    TaxID: <tax_id or unclear>
""".format(
            case_id=ex["case_id"],
            mention=ex["mention"],
            context=ex["context"],
        )

        out_p.write(prompt + "\n" + "="*100 + "\n")

        out_g.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
            ex["case_id"],
            ex["pmid"],
            ex["mention"],
            ex["gold_gene_ids"],
            ex["focus_taxid"],
            ex["pred_taxid"],
        ))

print("saved:", PROMPTS)
print("saved:", GOLD)
