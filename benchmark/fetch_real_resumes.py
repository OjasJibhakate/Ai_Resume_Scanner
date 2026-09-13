"""Fetch a real, publicly-licensed resume corpus for external validity checks.

**Why not scrape.** Resumes are dense with personal data - names, emails, phone
numbers, employers, dates. Scraping individuals' resumes for a research corpus
is a privacy violation regardless of whether the pages are reachable, and it
would fail ethics review at any venue worth submitting to. We use a published,
already-public research dataset instead.

**Dataset.** ``opensporks/resumes`` on the Hugging Face Hub - the livecareer.com
resume corpus widely used in the resume-screening literature (~2.4k resumes, 24
occupational categories). Critically for this project it ships ``Resume_str``,
which preserves line breaks, capitalisation and section headings. Corpora
preprocessed for bag-of-words classification (ResumeAtlas, for instance) are
lower-cased and stop-word-stripped, which destroys exactly the sentence and
section structure that evidence grounding reads - they are unusable here.

**Handling.** These are real documents about real people. This script caches the
text locally and nothing derived from it is redistributed. The analysis scripts
report aggregate distributions only and never print contact details. Treat
``data/real/`` as sensitive: it is gitignored.

Usage::

    python benchmark/fetch_real_resumes.py --limit 600
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "opensporks/resumes"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
#: Resumes are several KB each, and the rows endpoint drops the connection on
#: large pages, so we page in small batches with retries rather than asking for
#: a hundred multi-KB documents at once.
PAGE_SIZE = 20
MAX_RETRIES = 4

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)[\s-]?)?\d{3}[\s-]?\d{4}\b")
_URL = re.compile(r"https?://\S+|www\.\S+")


def redact(text: str) -> str:
    """Remove the most obvious direct identifiers.

    This is defence in depth, not anonymisation - free text can always carry
    identifying detail, and we make no claim that the result is anonymous. It
    exists so that routine debugging output cannot casually leak contact
    details, and because none of this pipeline needs them: emails and phone
    numbers contribute nothing to relevance.
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    return _PHONE.sub("[PHONE]", text)


def fetch_page(offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"{ROWS_ENDPOINT}?{query}", headers={"User-Agent": "talentlens-research"}
    )
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception as exc:  # connection resets are routine on this endpoint
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"rows request failed after {MAX_RETRIES} attempts: {last_error}")


def fetch(limit: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    offset = 0

    while len(records) < limit:
        want = min(PAGE_SIZE, limit - len(records))
        payload = fetch_page(offset, want)
        rows = payload.get("rows", [])
        if not rows:
            break
        for entry in rows:
            row = entry["row"]
            text = (row.get("Resume_str") or "").strip()
            if len(text) < 400:
                # Too short to carry a work narrative; would distort the
                # grounding distribution for reasons unrelated to grounding.
                continue
            records.append(
                {
                    "id": str(row.get("ID", f"row{offset}")),
                    "category": row.get("Category", "unknown"),
                    "text": redact(text),
                }
            )
        offset += want
        print(f"  fetched {len(records)} usable resumes (offset {offset})")

    path = out_dir / "real_resumes.json"
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")

    categories: dict[str, int] = {}
    for record in records:
        categories[record["category"]] = categories.get(record["category"], 0) + 1

    return {"count": len(records), "categories": categories, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", default=600, type=int)
    parser.add_argument("--out", default="data/real", type=Path)
    args = parser.parse_args()

    print(f"fetching up to {args.limit} resumes from {DATASET} ...")
    result = fetch(args.limit, args.out)
    print(f"\nsaved {result['count']} resumes to {result['path']}")
    top = sorted(result["categories"].items(), key=lambda kv: -kv[1])[:12]
    print("categories:", ", ".join(f"{k} ({v})" for k, v in top))
    print(
        "\nThese are real resumes about real people. Do not redistribute them, "
        "and do not print contact details in any output."
    )


if __name__ == "__main__":
    main()
