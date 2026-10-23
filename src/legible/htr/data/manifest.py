"""Line-level dataset manifests: one JSON object per line of a ``.jsonl`` file.

    {"image": "lines/a01-000u-00.png", "text": "A MOVE to stop", "writer": "000", "width": 1661, "height": 89}

``image`` is relative to the manifest's directory (absolute paths also work). ``writer``
lets evaluation splits stay writer-disjoint, which is the honest test for a model that
must read handwriting it has never seen. ``width``/``height`` are optional and only
used to batch similarly sized lines together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Record:
    image: str
    text: str
    writer: str | None = None
    width: int | None = None
    height: int | None = None

    def resolve(self, root: Path) -> Path:
        path = Path(self.image)
        return path if path.is_absolute() else root / path


def read_manifest(path: str | Path) -> list[Record]:
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(
                    Record(
                        image=str(data["image"]),
                        text=str(data["text"]),
                        writer=None if data.get("writer") is None else str(data["writer"]),
                        width=data.get("width"),
                        height=data.get("height"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: invalid manifest line ({exc})") from exc
    return records


def write_manifest(path: str | Path, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            data = {k: v for k, v in asdict(record).items() if v is not None}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count
