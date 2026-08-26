from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

ARROW = Path("nlm_gene-test.arrow")
OUT = Path("bioelqa_test_mentions.tsv")


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def get_text(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)


def build_full_text_by_offsets(doc):
    """
    Reconstruct full document text using passage offsets.
    This keeps entity offsets aligned with document text.
    """
    passages = doc.get("passages", [])
    max_end = 0

    for p in passages:
        offsets = p.get("offsets", [])
        if offsets:
            max_end = max(max_end, int(offsets[0][1]))

    chars = [" "] * max_end

    for p in passages:
        text = get_text(p.get("text", ""))
        offsets = p.get("offsets", [])
        if not offsets:
            continue

        start, end = offsets[0]
        start = int(start)
        end = int(end)

        for i, ch in enumerate(text[: max(0, end - start)]):
            pos = start + i
            if 0 <= pos < len(chars):
                chars[pos] = ch

    return "".join(chars)


def window_context(text, start, end, window=500):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].replace("\n", " ").strip()


def extract_gene_ids(ent):
    gene_ids = []

    norm_ids = ent.get("normalized", [])

    for n in norm_ids:
        db_id = n.get("db_id") or n.get("id")
        if db_id:
            gid = (
                str(db_id)
                .replace("NCBI Gene:", "")
                .replace("GeneID:", "")
                .replace("NCBIGene:", "")
                .strip()
            )
            if gid:
                gene_ids.append(gid)

    return sorted(set(gene_ids))


def main():
    rows = read_arrow(ARROW)

    records = []
    sanity_total = 0
    sanity_ok = 0

    for doc in rows:
        doc_id = doc.get("document_id", doc.get("id", ""))
        full_text = build_full_text_by_offsets(doc)

        for ent in doc.get("entities", []):
            mention = get_text(ent.get("text", ""))

            offsets = ent.get("offsets", [])
            if not offsets:
                continue

            start, end = offsets[0]
            start = int(start)
            end = int(end)

            gene_ids = extract_gene_ids(ent)

            if not gene_ids:
                continue

            span_text = full_text[start:end]
            sanity_total += 1
            if mention.lower() in span_text.lower() or span_text.lower() in mention.lower():
                sanity_ok += 1

            records.append({
                "doc_id": doc_id,
                "mention": mention,
                "start": start,
                "end": end,
                "gold_geneid": "|".join(gene_ids),
                "context": window_context(full_text, start, end, 500),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT, sep="\t", index=False)

    print("Saved:", OUT)
    print("Documents:", len(rows))
    print("Mentions:", len(df))
    print("Offset sanity checked:", sanity_total)
    print("Offset roughly matched mention:", sanity_ok)
    print("Offset match ratio:", round(sanity_ok / sanity_total, 4) if sanity_total else 0)

    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
