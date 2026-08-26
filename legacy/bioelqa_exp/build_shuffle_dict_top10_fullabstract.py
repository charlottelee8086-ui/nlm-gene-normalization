import json
import random
import sys
import re
import pandas as pd
from pathlib import Path

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1

PROMPTS_IN = Path(
    "bioelqa_dev_mcqa_prompts_dict_top10_fullabstract.jsonl"
)

GOLD_IN = Path(
    "bioelqa_dev_mcqa_gold_dict_top10_fullabstract.tsv"
)

PROMPTS_OUT = Path(
    f"bioelqa_dev_mcqa_prompts_dict_top10_fullabstract_shuffle_seed{SEED}.jsonl"
)

GOLD_OUT = Path(
    f"bioelqa_dev_mcqa_gold_dict_top10_fullabstract_shuffle_seed{SEED}.tsv"
)

MAP_OUT = Path(
    f"bioelqa_dev_mcqa_shuffle_map_seed{SEED}.tsv"
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def split_candidates(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [
        c.strip()
        for c in str(x).split("|")
        if c.strip()
    ]


def candidate_gid(c):
    return c.split("::", 1)[0].strip()


def prompt_gid(line):
    m = re.search(r"GeneID:\s*([^\s|]+)", line)
    if not m:
        raise ValueError(
            f"Could not extract GeneID from prompt line:\n{line}"
        )
    return m.group(1).strip()


# -------------------------------------------------------
# Read gold TSV
# -------------------------------------------------------

gold = pd.read_csv(GOLD_IN, sep="\t")

gold_by_case = {
    str(r["case_id"]): r
    for _, r in gold.iterrows()
}


# -------------------------------------------------------
# Read prompts
# -------------------------------------------------------

prompts = []

with open(PROMPTS_IN, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            prompts.append(json.loads(line))


print("Prompts:", len(prompts))
print("Gold rows:", len(gold))

assert len(prompts) == len(gold)


# -------------------------------------------------------
# Shuffle
# -------------------------------------------------------

rng = random.Random(SEED)

new_prompts = []
new_gold_rows = []
map_rows = []

alignment_ok = 0


for obj in prompts:

    case_id = str(obj["case_id"])
    prompt = obj["prompt"]

    if case_id not in gold_by_case:
        raise ValueError(f"Missing gold row: {case_id}")

    row = gold_by_case[case_id]

    candidates = split_candidates(row["candidates"])

    # ---------------------------------------------------
    # Find candidate block in prompt
    # ---------------------------------------------------

    marker1 = "Candidates:\n"
    marker2 = "\n\nOnly output:"

    if marker1 not in prompt or marker2 not in prompt:
        raise ValueError(
            f"Candidate block not found for {case_id}"
        )

    before, rest = prompt.split(marker1, 1)
    cand_text, after = rest.split(marker2, 1)

    option_lines = [
        x for x in cand_text.splitlines()
        if x.strip()
    ]

    if len(option_lines) != len(candidates):
        raise ValueError(
            f"{case_id}: prompt candidates={len(option_lines)} "
            f"TSV candidates={len(candidates)}"
        )

    # ---------------------------------------------------
    # Verify prompt order == TSV candidate order
    # ---------------------------------------------------

    prompt_gids = [
        prompt_gid(x)
        for x in option_lines
    ]

    tsv_gids = [
        candidate_gid(x)
        for x in candidates
    ]

    if prompt_gids != tsv_gids:
        print("\nAlignment failure:", case_id)
        print("Prompt:", prompt_gids)
        print("TSV:", tsv_gids)
        raise ValueError("Candidate alignment failed")

    alignment_ok += 1

    # ---------------------------------------------------
    # Generate one random permutation
    # ---------------------------------------------------

    indices = list(range(len(candidates)))
    rng.shuffle(indices)

    shuffled_candidates = [
        candidates[i]
        for i in indices
    ]

    shuffled_lines = [
        option_lines[i]
        for i in indices
    ]

    # Rename option letters A, B, C ...
    relabeled_lines = []

    for new_pos, line in enumerate(shuffled_lines):

        new_letter = LETTERS[new_pos]

        line = re.sub(
            r"^[A-Z]\.",
            f"{new_letter}.",
            line.strip()
        )

        relabeled_lines.append(line)

    # ---------------------------------------------------
    # Rebuild prompt
    # ---------------------------------------------------

    new_prompt = (
        before
        + marker1
        + "\n".join(relabeled_lines)
        + marker2
        + after
    )

    new_obj = dict(obj)
    new_obj["prompt"] = new_prompt

    new_prompts.append(new_obj)

    # ---------------------------------------------------
    # Rebuild gold TSV row with SAME permutation
    # ---------------------------------------------------

    new_row = row.copy()
    new_row["candidates"] = "|".join(shuffled_candidates)

    new_gold_rows.append(new_row)

    # ---------------------------------------------------
    # Save permutation for later consistency analysis
    # ---------------------------------------------------

    for new_pos, old_pos in enumerate(indices):

        map_rows.append({
            "case_id": case_id,
            "new_option": LETTERS[new_pos],
            "old_option": LETTERS[old_pos],
            "geneid": candidate_gid(candidates[old_pos]),
        })


# -------------------------------------------------------
# Write files
# -------------------------------------------------------

with open(PROMPTS_OUT, "w", encoding="utf-8") as f:
    for obj in new_prompts:
        f.write(
            json.dumps(obj, ensure_ascii=False)
            + "\n"
        )


pd.DataFrame(new_gold_rows).to_csv(
    GOLD_OUT,
    sep="\t",
    index=False
)


pd.DataFrame(map_rows).to_csv(
    MAP_OUT,
    sep="\t",
    index=False
)


print("\n=== Done ===")
print(
    "Prompt/TSV alignment:",
    alignment_ok,
    "/",
    len(prompts)
)

print("Saved:", PROMPTS_OUT)
print("Saved:", GOLD_OUT)
print("Saved:", MAP_OUT)
