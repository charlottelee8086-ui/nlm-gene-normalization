# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

ARROW = Path("nlm_gene-train.arrow")
TRAIN_TSV = Path("bioelqa_train_mentions.tsv")
DEV_TSV = Path("bioelqa_dev_mentions.tsv")

OUT_TRAIN = Path("bioelqa_train_mentions_contexts.tsv")
OUT_DEV = Path("bioelqa_dev_mentions_contexts.tsv")


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


def get_passage_text(doc, ptype):
    parts = []
    for p in doc.get("passages", []):
        if str(p.get("type", "")).lower() == ptype:
            parts.append(get_text(p.get("text", "")))
    return " ".join(parts).strip()


def build_full_text_by_offsets(doc):
    """
    Reconstruct document text using passage offsets.
    This keeps entity offsets aligned with the document text.
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


def sentence_spans(text):
    """
    Simple sentence splitter.
    It returns: [(start_char, end_char, sentence_text), ...]
    """
    spans = []

    for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
        s = m.start()
        e = m.end()
        raw = text[s:e]

        sent = raw.strip()
        if not sent:
            continue

        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())

        spans.append((s + leading, e - trailing, sent))

    return spans


def find_sentence_index(spans, start, end):
    for i, (s, e, sent) in enumerate(spans):
        # overlap between mention span and sentence span
        if max(s, start) < min(e, end):
            return i
    return None


def window_context(text, start, end, window=500):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].replace("\n", " ").strip()


def add_contexts(mention_df, doc_map, split_name):
    out_rows = []
    sanity_total = 0
    sanity_ok = 0

    for _, r in mention_df.iterrows():
        doc_id = str(r["doc_id"])
        mention = str(r["mention"])
        start = int(r["start"])
        end = int(r["end"])

        info = doc_map.get(doc_id)

        row = r.to_dict()
        row["ctx_mention"] = mention

        if info is None:
            row["ctx_sentence"] = ""
            row["ctx_3sent"] = ""
            row["ctx_500"] = str(r.get("context", ""))
            row["ctx_abstract"] = ""
            row["ctx_document"] = ""
            out_rows.append(row)
            continue

        full_text = info["full_text"]
        spans = info["sentence_spans"]

        # offset sanity check
        sanity_total += 1
        span_text = full_text[start:end]
        if mention.lower() in span_text.lower() or span_text.lower() in mention.lower():
            sanity_ok += 1

        sent_idx = find_sentence_index(spans, start, end)

        if sent_idx is not None:
            ctx_sentence = spans[sent_idx][2]

            left_idx = max(0, sent_idx - 1)
            right_idx = min(len(spans), sent_idx + 2)
            ctx_3sent = " ".join(spans[i][2] for i in range(left_idx, right_idx))
        else:
            # fallback if sentence splitting fails
            ctx_sentence = window_context(full_text, start, end, window=120)
            ctx_3sent = window_context(full_text, start, end, window=250)

        row["ctx_sentence"] = ctx_sentence
        row["ctx_3sent"] = ctx_3sent
        row["ctx_500"] = window_context(full_text, start, end, window=500)
        row["ctx_abstract"] = info["abstract_text"]
        row["ctx_document"] = info["document_text"]

        out_rows.append(row)

    print("=" * 80)
    print(split_name)
    print("Rows:", len(mention_df))
    print("Offset sanity checked:", sanity_total)
    print("Offset roughly matched mention:", sanity_ok)
    print("Offset match ratio:", round(sanity_ok / sanity_total, 4) if sanity_total else 0)

    return pd.DataFrame(out_rows)


def main():
    print("Reading arrow...")
    rows = read_arrow(ARROW)

    doc_map = {}

    for doc in rows:
        doc_id = str(doc.get("document_id", doc.get("id", "")))

        title = get_passage_text(doc, "title")
        abstract = get_passage_text(doc, "abstract")
        document_text = (title + " " + abstract).strip()

        full_text = build_full_text_by_offsets(doc)

        doc_map[doc_id] = {
            "title_text": title,
            "abstract_text": abstract,
            "document_text": document_text,
            "full_text": full_text,
            "sentence_spans": sentence_spans(full_text),
        }

    print("Documents loaded:", len(doc_map))

    print("Reading mention TSV files...")
    train = pd.read_csv(TRAIN_TSV, sep="\t")
    dev = pd.read_csv(DEV_TSV, sep="\t")

    train_out = add_contexts(train, doc_map, "TRAIN")
    dev_out = add_contexts(dev, doc_map, "DEV")

    train_out.to_csv(OUT_TRAIN, sep="\t", index=False)
    dev_out.to_csv(OUT_DEV, sep="\t", index=False)

    print("=" * 80)
    print("Saved:", OUT_TRAIN)
    print("Saved:", OUT_DEV)

    print("\nOutput columns:")
    print(dev_out.columns.tolist())

    print("\nSample dev rows:")
    show_cols = [
        "doc_id",
        "mention",
        "gold_geneid",
        "ctx_sentence",
        "ctx_3sent",
        "ctx_500",
    ]
    print(dev_out[show_cols].head(3).to_string())


if __name__ == "__main__":
    main()
