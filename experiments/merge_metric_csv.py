from __future__ import annotations

import argparse
import csv
from pathlib import Path


def merge_csv(inputs: list[Path], output: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for input_path in inputs:
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"CSV columns differ in {input_path}")
            rows.extend(reader)
    if fieldnames is None:
        raise ValueError("at least one input CSV is required")
    rows.sort(key=lambda row: (int(row["seed"]), row["tracker"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_csv(args.input, args.output)


if __name__ == "__main__":
    main()
