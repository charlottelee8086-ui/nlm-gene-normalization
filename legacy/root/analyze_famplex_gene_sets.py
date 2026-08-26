from collections import Counter

FILE = "missing_famplex_matches.tsv"

seen = set()

single = 0
multi = 0

counter = Counter()

with open(FILE, encoding="utf-8") as f:
    next(f)

    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 8:
            continue

        key = tuple(parts[:5])

        if key in seen:
            continue

        seen.add(key)

        gids = parts[4]

        n = len(gids.split("|"))

        if n == 1:
            single += 1
        else:
            multi += 1

        counter[n] += 1

print("Unique FamPlex-covered mentions:", len(seen))
print("Single Gene:", single)
print("Multi Gene:", multi)

print("\nDistribution")

for k in sorted(counter):
    print(k, counter[k])
