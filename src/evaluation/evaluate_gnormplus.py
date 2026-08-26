#!/usr/bin/env python3

import argparse

import pyarrow as pa
import pyarrow.ipc as ipc


GENE_TYPES = {
    "Gene",
    "GENERIF",
    "STARGENE",
}


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def normalize_geneid(value):
    return (
        str(value)
        .replace("NCBIGene:", "")
        .replace("*", "")
        .strip()
    )


def load_gold_by_span(path):
    """
    Map each annotated mention span to all acceptable NCBI GeneIDs.
    """
    gold = {}

    for document in read_arrow(path):
        document_id = str(document["document_id"])

        for entity in document["entities"]:
            if entity["type"] not in GENE_TYPES:
                continue

            normalized = entity.get("normalized")

            if not normalized:
                continue

            start, end = entity["offsets"][0]

            gene_ids = {
                str(item["db_id"])
                for item in normalized
                if item.get("db_name") == "NCBIGene"
            }

            if gene_ids:
                gold[
                    (
                        document_id,
                        int(start),
                        int(end),
                    )
                ] = gene_ids

    return gold


def load_predictions_by_span(path):
    """
    Map each GNormPlus gene span to its predicted GeneID.
    """
    predictions = {}

    with open(
        path,
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 6:
                continue

            (
                document_id,
                start,
                end,
                mention,
                entity_type,
                gene_id,
            ) = parts[:6]

            if entity_type != "Gene":
                continue

            gene_id = normalize_geneid(gene_id)

            if not gene_id or not gene_id[0].isdigit():
                continue

            predictions[
                (
                    str(document_id),
                    int(start),
                    int(end),
                )
            ] = gene_id

    return predictions


def evaluate_mention_full_test(gold_path, prediction_path):
    """
    Full-test accuracy used for comparison with the other
    normalization methods.

    Denominator: every gold NLM-Gene mention.
    """
    gold = load_gold_by_span(gold_path)
    predictions = load_predictions_by_span(prediction_path)

    recognized = 0
    correct = 0
    wrong = 0
    missing = 0

    for span, gold_geneids in gold.items():
        if span not in predictions:
            missing += 1
            continue

        recognized += 1

        if predictions[span] in gold_geneids:
            correct += 1
        else:
            wrong += 1

    total = len(gold)

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    print("=== Mention-level full-test evaluation ===")
    print("Gold mentions:", total)
    print("Recognized gold spans:", recognized)
    print("Correct links:", correct)
    print("Wrong links:", wrong)
    print("Missing gold mentions:", missing)
    print()
    print(
        f"Full-test accuracy: "
        f"{accuracy:.4f} ({accuracy:.2%})"
    )


def evaluate_belb_style(gold_path, prediction_path):
    """
    Historical BELB-style linking evaluation.

    Denominator: only gold spans recognized by GNormPlus.
    """
    gold = load_gold_by_span(gold_path)
    predictions = load_predictions_by_span(prediction_path)

    recognized = 0
    correct = 0
    wrong = 0

    for span, gold_geneids in gold.items():
        if span not in predictions:
            continue

        recognized += 1

        if predictions[span] in gold_geneids:
            correct += 1
        else:
            wrong += 1

    accuracy = (
        correct / recognized
        if recognized
        else 0.0
    )

    recognition_rate = (
        recognized / len(gold)
        if gold
        else 0.0
    )

    print("=== BELB-style linking evaluation ===")
    print("Gold mentions:", len(gold))
    print("Predicted gene mentions:", len(predictions))
    print("Recognized gold spans:", recognized)
    print("Correct links:", correct)
    print("Wrong links:", wrong)
    print()
    print(
        f"Gold-span recognition rate: "
        f"{recognition_rate:.2%}"
    )
    print(
        f"Linking accuracy on recognized spans: "
        f"{accuracy:.4f} ({accuracy:.2%})"
    )


def load_gold_tuples(path):
    gold = set()

    for document in read_arrow(path):
        document_id = str(document["document_id"])

        for entity in document["entities"]:
            if entity["type"] not in GENE_TYPES:
                continue

            normalized = entity.get("normalized")

            if not normalized:
                continue

            start, end = entity["offsets"][0]

            for item in normalized:
                if item.get("db_name") != "NCBIGene":
                    continue

                gold.add(
                    (
                        document_id,
                        int(start),
                        int(end),
                        str(item["db_id"]),
                    )
                )

    return gold


def load_prediction_tuples(path):
    predictions = set()

    with open(
        path,
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 6:
                continue

            (
                document_id,
                start,
                end,
                mention,
                entity_type,
                gene_id,
            ) = parts[:6]

            if entity_type != "Gene":
                continue

            predictions.add(
                (
                    str(document_id),
                    int(start),
                    int(end),
                    str(gene_id),
                )
            )

    return predictions


def evaluate_tuple_prf(gold_path, prediction_path):
    """
    Reproduce the historical eval_gnorm2.py calculation.

    This treats every acceptable GeneID annotation as a separate
    (document, start, end, GeneID) tuple.
    """
    gold = load_gold_tuples(gold_path)
    predictions = load_prediction_tuples(prediction_path)

    tp = len(gold & predictions)
    fp = len(predictions - gold)
    fn = len(gold - predictions)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print("=== Tuple-level precision / recall / F1 ===")
    print("Gold tuples:", len(gold))
    print("Predicted tuples:", len(predictions))
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print()
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall:    {recall:.4f} ({recall:.2%})")
    print(f"F1:        {f1:.4f} ({f1:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNormPlus outputs on NLM-Gene."
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="NLM-Gene Arrow test file.",
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="GNormPlus PubTator output.",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "mention-full-test",
            "belb-style",
            "tuple-prf",
        ],
    )

    args = parser.parse_args()

    if args.mode == "mention-full-test":
        evaluate_mention_full_test(
            args.gold,
            args.predictions,
        )

    elif args.mode == "belb-style":
        evaluate_belb_style(
            args.gold,
            args.predictions,
        )

    else:
        evaluate_tuple_prf(
            args.gold,
            args.predictions,
        )


if __name__ == "__main__":
    main()
