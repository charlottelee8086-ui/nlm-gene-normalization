# -*- coding: utf-8 -*-

import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from collections import defaultdict

ARROW = Path("nlm_gene-test.arrow")
ORIG_TMP_SA = Path.home() / "nlm_gene_repro/GNorm2/tmp_SA/nlm_gene_test.PubTator"
OUT = Path("gold_tmp_SA.PubTator")


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def make_text_lines(doc):
    pmid = str(doc["document_id"])
    passages = sorted(doc["passages"], key=lambda p: p["offsets"][0][0])

    lines = []

    for i, p in enumerate(passages):
        text = p["text"][0]

        if isinstance(text, bytes):
            text = text.decode("utf-8")

        if i == 0:
            lines.append("{}|t|{}\n".format(pmid, text))
        else:
            lines.append("{}|a|{}\n".format(pmid, text))

    return lines


# Get species focus from original GNormPlus tmp_SA
pmid2focus = defaultdict(lambda: "9606")

with open(str(ORIG_TMP_SA), "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "\tSpecies\t*" not in line:
            continue

        parts = line.rstrip("\n").split("\t")

        if len(parts) >= 6:
            pmid = parts[0]
            taxid = parts[-1].replace("*", "").strip()

            if taxid:
                pmid2focus[pmid] = taxid


rows = read_arrow(ARROW)

with open(str(OUT), "w", encoding="utf-8") as out:
    for doc in rows:
        pmid = str(doc["document_id"])

        for line in make_text_lines(doc):
            out.write(line)

        taxid = pmid2focus[pmid]

        out.write("{}\t0\t0\t{}\tSpecies\t*{}\n".format(
            pmid,
            taxid,
            taxid,
        ))

        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            mention = ent["text"][0].strip()

            if isinstance(mention, bytes):
                mention = mention.decode("utf-8")

            start, end = ent["offsets"][0]

            out.write("{}\t{}\t{}\t{}\tGene\tFocus:{}\n".format(
                pmid,
                int(start),
                int(end),
                mention,
                taxid,
            ))

        out.write("\n")

print("saved: {}".format(OUT))
