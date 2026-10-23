"""Turns raw handwriting datasets into line manifests (see :mod:`.data.manifest`).

    # a folder of line images with same-name .txt transcriptions
    legible-prepare folder path/to/lines --out data/mine/all.jsonl

    # a Hugging Face dataset with image + text columns (needs `pip install -e .[train]`)
    legible-prepare hf Teklia/IAM-line --out data/iam

    # writer-disjoint train/validation/test split of a manifest
    legible-prepare split data/mine/all.jsonl --val 0.1 --test 0.1

    # plain-text corpus for synthetic lines (normalized, de-duplicated)
    legible-prepare corpus texts/*.txt --out data/corpus/sentences.txt

Respect each dataset's license; several handwriting datasets (including IAM) are for
non-commercial research only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data.manifest import Record, read_manifest, write_manifest
from .vocab import normalize_text

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def from_folder(folder: Path, out: Path, writer_from_parent: bool = False) -> int:
    records = []
    for image in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        label = image.with_suffix(".txt")
        if not label.exists():
            continue
        text = normalize_text(label.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue
        with Image.open(image) as img:
            width, height = img.size
        writer = image.parent.name if writer_from_parent else None
        records.append(Record(_relative(image, out.parent), text, writer, width, height))
    count = write_manifest(out, records)
    log.info("wrote %d lines to %s", count, out)
    return count


def from_huggingface(name: str, out: Path, image_column: str, text_column: str, writer_column: str | None,
                     splits: list[str] | None) -> dict[str, int]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("the `datasets` package is needed: pip install -e '.[train]'") from None
    dataset = load_dataset(name)
    counts = {}
    for split in splits or list(dataset.keys()):
        rows = dataset[split]
        folder = out / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for i, row in enumerate(rows):
            text = normalize_text(str(row[text_column]))
            if not text:
                continue
            image: Image.Image = row[image_column]
            path = folder / f"{i:07d}.png"
            if not path.exists():
                image.convert("L").save(path)
            writer = str(row[writer_column]) if writer_column and writer_column in row else None
            records.append(Record(_relative(path, out), text, writer, image.width, image.height))
        counts[split] = write_manifest(out / f"{split}.jsonl", records)
        log.info("%s: %d lines", split, counts[split])
    return counts


def split_manifest(manifest: Path, val: float, test: float, seed: int) -> dict[str, int]:
    """Writer-disjoint split when writers are known (otherwise by line)."""
    records = read_manifest(manifest)
    groups: dict[str, list[Record]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[record.writer or f"line-{i}"].append(record)
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    total = len(records)
    parts: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    for key in keys:
        if len(parts["test"]) < test * total:
            parts["test"].extend(groups[key])
        elif len(parts["validation"]) < val * total:
            parts["validation"].extend(groups[key])
        else:
            parts["train"].extend(groups[key])
    counts = {}
    for name, items in parts.items():
        counts[name] = write_manifest(manifest.with_name(f"{name}.jsonl"), items)
    log.info("split %s: %s", manifest, counts)
    return counts


def build_corpus(inputs: list[Path], out: Path, min_chars: int, max_chars: int) -> int:
    seen: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for path in inputs:
            with path.open(encoding="utf-8", errors="ignore") as src:
                for line in src:
                    text = normalize_text(line)
                    if min_chars <= len(text) <= max_chars and text not in seen:
                        seen.add(text)
                        f.write(text + "\n")
    log.info("wrote %d corpus lines to %s", len(seen), out)
    return len(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare handwriting datasets as line manifests.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("folder", help="images with same-name .txt transcriptions")
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--writer-from-parent", action="store_true", help="use each image's folder name as the writer id")

    p = sub.add_parser("hf", help="a Hugging Face dataset with image and text columns")
    p.add_argument("name")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-column", default="image")
    p.add_argument("--text-column", default="text")
    p.add_argument("--writer-column", default=None)
    p.add_argument("--splits", nargs="*")

    p = sub.add_parser("split", help="writer-disjoint train/validation/test split")
    p.add_argument("manifest", type=Path)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("corpus", help="normalized, de-duplicated text for synthetic lines")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-chars", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=200)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "folder":
        from_folder(args.folder, args.out, args.writer_from_parent)
    elif args.command == "hf":
        from_huggingface(args.name, args.out, args.image_column, args.text_column, args.writer_column, args.splits)
    elif args.command == "split":
        split_manifest(args.manifest, args.val, args.test, args.seed)
    else:
        build_corpus(args.inputs, args.out, args.min_chars, args.max_chars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
