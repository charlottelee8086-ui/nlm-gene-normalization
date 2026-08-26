from pathlib import Path
import csv
import re
from collections import defaultdict, Counter

MISSING = Path("gold_missing_normalization.tsv")
FAMPLEX_DIR = Path.home() / "famplex"
OUT = Path("missing_famplex_matches.tsv")


def norm_text(s):
    s = s.strip().lower()
    s = s.replace("κ", "kappa")
    s = s.replace("β", "beta")
    s = s.replace("α", "alpha")
    s = s.replace("γ", "gamma")
    s = s.replace("δ", "delta")
    s = re.sub(r"[\s_\-/]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def read_table(path):
    # Try tab first for .tsv, comma for .csv
    if path.suffix.lower() == ".tsv":
        delimiter = "\t"
    else:
        delimiter = ","

    rows = []
    try:
        with open(path, encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if row:
                    rows.append(row)
    except Exception:
        return []

    return rows


# Load missing mentions
missing_rows = []

with open(MISSING, encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        row = dict(zip(header, parts))
        row["mention_norm"] = norm_text(row["mention"])
        missing_rows.append(row)

missing_norms = set(r["mention_norm"] for r in missing_rows)


# Scan FamPlex files broadly
matches = defaultdict(list)

candidate_files = []
for ext in ["*.csv", "*.tsv", "*.txt"]:
    candidate_files.extend(FAMPLEX_DIR.rglob(ext))

for path in candidate_files:
    rows = read_table(path)
    if not rows:
        continue

    for i, row in enumerate(rows):
        for cell in row:
            cell_norm = norm_text(cell)
            if cell_norm in missing_norms:
                matches[cell_norm].append({
                    "file": str(path),
                    "line": i + 1,
                    "row": row,
                })


# Write matches
with open(OUT, "w", encoding="utf-8") as out:
    out.write("pmid\tstart\tend\tmention\tgold_gene_ids\tfamplex_file\tfamplex_line\tfamplex_row\n")

    hit_count = 0

    for row in missing_rows:
        mnorm = row["mention_norm"]

        if mnorm not in matches:
            continue

        hit_count += 1

        for hit in matches[mnorm][:5]:
            out.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    row["pmid"],
                    row["start"],
                    row["end"],
                    row["mention"],
                    row["gold_gene_ids"],
                    hit["file"],
                    hit["line"],
                    " | ".join(hit["row"]),
                )
            )

print("missing mentions:", len(missing_rows))
print("unique missing mention strings:", len(set(r["mention"] for r in missing_rows)))
print("FamPlex matched mentions:", hit_count)
print("saved:", OUT)

# frequency of matched mentions
counter = Counter()
for row in missing_rows:
    if row["mention_norm"] in matches:
        counter[row["mention"]] += 1

print("\n=== Top FamPlex-covered missing mentions ===\n")
for mention, freq in counter.most_common(50):
    print(freq, mention)
