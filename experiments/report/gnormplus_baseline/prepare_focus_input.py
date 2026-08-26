#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


GENE_TYPES = {
    "Gene",
    "GENERIF",
    "STARGENE",
}


def read_arrow(path):
    with pa.memory_map(
        str(path),
        "r",
    ) as source:
        try:
            reader = ipc.RecordBatchFileReader(
                source
            )
            return reader.read_all().to_pylist()

        except Exception:
            source.seek(0)

            reader = ipc.RecordBatchStreamReader(
                source
            )

            return reader.read_all().to_pylist()


def load_species_focus(path):
    focus = defaultdict(
        lambda: "9606"
    )

    with open(
        path,
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            if "\tSpecies\t*" not in line:
                continue

            parts = line.rstrip(
                "\n"
            ).split("\t")

            if len(parts) < 6:
                continue

            document_id = parts[0]

            tax_id = (
                parts[-1]
                .replace("*", "")
                .strip()
            )

            if tax_id:
                focus[
                    document_id
                ] = tax_id

    return focus


def text_lines(document):
    document_id = str(
        document["document_id"]
    )

    passages = sorted(
        document["passages"],
        key=lambda passage: (
            passage["offsets"][0][0]
        ),
    )

    lines = []

    for index, passage in enumerate(
        passages
    ):
        text = passage["text"][0]

        if isinstance(
            text,
            bytes,
        ):
            text = text.decode(
                "utf-8"
            )

        section = (
            "t"
            if index == 0
            else "a"
        )

        lines.append(
            f"{document_id}|{section}|{text}\n"
        )

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a PubTator focus file for the historical GNormPlus experiment."
    )

    parser.add_argument(
        "--nlm-gene",
        required=True,
        help="NLM-Gene Arrow file.",
    )

    parser.add_argument(
        "--species-assignment",
        required=True,
        help="GNormPlus tmp_SA file containing document-level species assignments.",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    rows = read_arrow(
        args.nlm_gene
    )

    document_to_taxid = load_species_focus(
        args.species_assignment
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for document in rows:
            document_id = str(
                document["document_id"]
            )

            for line in text_lines(
                document
            ):
                output_file.write(
                    line
                )

            tax_id = document_to_taxid[
                document_id
            ]

            output_file.write(
                f"{document_id}\t0\t0\t"
                f"{tax_id}\tSpecies\t*{tax_id}\n"
            )

            for entity in document[
                "entities"
            ]:
                if (
                    entity["type"]
                    not in GENE_TYPES
                ):
                    continue

                mention = entity[
                    "text"
                ][0]

                if isinstance(
                    mention,
                    bytes,
                ):
                    mention = mention.decode(
                        "utf-8"
                    )

                mention = mention.strip()

                start, end = entity[
                    "offsets"
                ][0]

                output_file.write(
                    f"{document_id}\t"
                    f"{int(start)}\t"
                    f"{int(end)}\t"
                    f"{mention}\t"
                    f"Gene\t"
                    f"Focus:{tax_id}\n"
                )

            output_file.write(
                "\n"
            )

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
