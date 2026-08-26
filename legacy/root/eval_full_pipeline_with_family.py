TOTAL_GOLD = 2729

# From eval_gold_normalization_with_rescue.py
BASE_GNORMPLUS_CORRECT = 1489
RESCUE_FINAL_CORRECT = 1705

# From family subset experiments
FAMILY_V5_CORRECT = 85
FAMILY_V6_CORRECT = 108
FAMILY_V7_AUTO_QWEN14_CORRECT = 97

systems = [
    ("GNormPlus", BASE_GNORMPLUS_CORRECT),
    ("GNormPlus + synonym rescue", RESCUE_FINAL_CORRECT),
    ("+ family reranker v5", RESCUE_FINAL_CORRECT + FAMILY_V5_CORRECT),
    ("+ family LLM/Qwen hybrid v6", RESCUE_FINAL_CORRECT + FAMILY_V6_CORRECT),
    ("+ auto Qwen14 family trigger v7", RESCUE_FINAL_CORRECT + FAMILY_V7_AUTO_QWEN14_CORRECT),
]

print("System\tCorrect\tTotal\tAccuracy")

for name, correct in systems:
    print(
        "{}\t{}\t{}\t{:.6f}".format(
            name,
            correct,
            TOTAL_GOLD,
            correct / TOTAL_GOLD,
        )
    )
