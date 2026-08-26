from collections import Counter

FILE = "gold_missing_normalization.tsv"

single = 0
multi = 0

counter = Counter()

with open(FILE, encoding="utf-8") as f:
    header = next(f)

    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 5:
            continue

        gids = parts[4]

        n = len(gids.split("|"))

        if n == 1:
            single += 1
        else:
            multi += 1

        counter[n] += 1

print("Single Gene:", single)
print("Multi Gene:", multi)

print("\nDistribution")

for k in sorted(counter):
    print(k, counter[k])

