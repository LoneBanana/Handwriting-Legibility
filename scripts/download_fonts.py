#!/usr/bin/env python3
"""Downloads open-licensed handwriting fonts for synthetic training data.

    python scripts/download_fonts.py                 # -> assets/fonts/
    python scripts/download_fonts.py --out /scratch/fonts

The families below are on Google Fonts under the SIL Open Font License (or Apache 2.0),
which allows using them to render training images. They span neat school print
(the Edu and Playpen families), casual print, marker, and cursive styles, so the
model sees many letter forms. The script asks the Google Fonts CSS API for each family
and saves the TrueType files it links to, plus a SOURCES.txt listing where each came
from. Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

FAMILIES = [
    # school-style print and early-handwriting fonts
    "Edu SA Beginner", "Edu NSW ACT Foundation", "Edu QLD Beginner", "Edu VIC WA NT Beginner", "Playpen Sans",
    "Schoolbell", "Patrick Hand", "Handlee", "Delius", "Itim", "Mali", "Short Stack", "Comic Neue",
    # casual handwriting
    "Caveat", "Indie Flower", "Shadows Into Light", "Gochi Hand", "Kalam", "Architects Daughter",
    "Covered By Your Grace", "Reenie Beanie", "Just Another Hand", "Sue Ellen Francisco", "Gloria Hallelujah",
    "Neucha", "Coming Soon", "Annie Use Your Telescope", "The Girl Next Door", "Loved by the King",
    "Give You Glory", "Crafty Girls", "Swanky and Moo Moo", "Over the Rainbow", "Waiting for the Sunrise",
    "Nothing You Could Do", "Gaegu", "Gamja Flower", "Hi Melody", "Nanum Pen Script", "Mynerve", "Grape Nuts",
    "Shantell Sans", "Sriracha", "Amatic SC", "Permanent Marker", "Rock Salt", "Caveat Brush",
    # cursive and joined handwriting
    "Homemade Apple", "La Belle Aurore", "Dawning of a New Day", "Zeyada", "Cedarville Cursive",
    "Mrs Saint Delafield", "Kristi", "Satisfy", "Dancing Script", "Sacramento", "Marck Script", "Square Peg",
]

CSS_API = "https://fonts.googleapis.com/css2?family={family}"
FONT_URL = re.compile(r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)")


def fetch(url: str, timeout: float = 30.0) -> bytes:
    # A plain (non-browser) user agent makes the CSS API link TrueType files.
    request = urllib.request.Request(url, headers={"User-Agent": "legible-font-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("assets/fonts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = []
    failed = []
    for family in FAMILIES:
        slug = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")
        try:
            css = fetch(CSS_API.format(family=urllib.parse.quote_plus(family))).decode("utf-8")
            urls = list(dict.fromkeys(FONT_URL.findall(css)))
            if not urls:
                raise RuntimeError("no TrueType link in the CSS response")
            for i, url in enumerate(urls):
                path = args.out / f"{slug}{'' if i == 0 else f'-{i}'}.ttf"
                if not path.exists():
                    path.write_bytes(fetch(url))
                sources.append(f"{path.name}\t{family}\t{url}")
            print(f"ok   {family}")
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failed.append(family)
            print(f"FAIL {family}: {exc}", file=sys.stderr)
        time.sleep(0.2)
    (args.out / "SOURCES.txt").write_text(
        "Fonts from Google Fonts (SIL Open Font License 1.1 or Apache 2.0); see fonts.google.com.\n"
        + "\n".join(sources) + "\n"
    )
    print(f"\n{len(sources)} font files in {args.out}" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 0 if sources else 1


if __name__ == "__main__":
    sys.exit(main())
