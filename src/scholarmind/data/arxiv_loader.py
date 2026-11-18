"""ArXiv metadata loader supporting manual Kaggle JSONL streaming and lightweight API sampling."""

from __future__ import annotations

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

KAGGLE_ARXIV_URL = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
ARXIV_API_ENDPOINT = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def get_kaggle_manual_instructions(target_path: Path) -> str:
    """Return clear instructions for manually downloading the Kaggle ArXiv metadata snapshot."""
    return (
        "\n========================================================================\n"
        "MANUAL KAGGLE DOWNLOAD REQUIRED FOR FULL ARXIV METADATA SNAPSHOT\n"
        "========================================================================\n"
        f"1. Visit: {KAGGLE_ARXIV_URL}\n"
        "2. Sign in to Kaggle and download 'arxiv-metadata-oai-snapshot.json' (~3+ GB).\n"
        f"3. Extract and place the JSONL file at:\n"
        f"   {target_path}\n"
        "   (Or set ARXIV_KAGGLE_JSON_PATH in your .env file.)\n"
        "4. Alternatively, for immediate lightweight development & testing without\n"
        "   downloading 3+ GB, run:\n"
        "   python scripts/download_datasets.py --datasets arxiv --arxiv-source api --scale mini\n"
        "========================================================================\n"
    )


def _extract_year(record: dict) -> int | None:
    """Extract publication year from Kaggle arXiv `versions` or `update_date` fields."""
    versions = record.get("versions")
    if isinstance(versions, list) and versions:
        created = str(versions[0].get("created", ""))
        for token in created.split():
            if len(token) == 4 and token.isdigit():
                year = int(token)
                if 1985 <= year <= 2030:
                    return year
    update_date = str(record.get("update_date") or record.get("published") or "")
    if len(update_date) >= 4 and update_date[:4].isdigit():
        return int(update_date[:4])
    return None


def load_arxiv_jsonl(
    jsonl_path: Path,
    target_categories: Iterable[str] | None = None,
    max_records: int | None = 5000,
    min_abstract_words: int = 20,
) -> pd.DataFrame:
    """Stream-parse Kaggle ArXiv JSONL line-by-line without loading the 3+ GB file into RAM.

    Supports both the official Kaggle `arxiv-metadata-oai-snapshot.json` schema and
    the compatible `arxiv-sample-api.jsonl` format produced by `fetch_arxiv_api_sample`.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"ArXiv JSONL file not found at {jsonl_path}.\n"
            + get_kaggle_manual_instructions(jsonl_path)
        )

    target_set = set(target_categories) if target_categories else None
    rows: list[dict] = []
    scanned = 0
    malformed = 0

    logger.info("Streaming ArXiv JSONL from %s (max_records=%s)...", jsonl_path, max_records)
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            raw_cats = str(rec.get("categories") or "").strip()
            cat_list = [c for c in raw_cats.split() if c]
            if not cat_list:
                continue

            if target_set is not None:
                matched_cats = [c for c in cat_list if c in target_set]
                if not matched_cats:
                    continue
                primary_cat = matched_cats[0]
            else:
                matched_cats = cat_list
                primary_cat = cat_list[0]

            abstract = str(rec.get("abstract") or "").strip()
            title = str(rec.get("title") or "").strip()
            if len(abstract.split()) < min_abstract_words or not title:
                continue

            rows.append(
                {
                    "paper_id": str(rec.get("id") or f"arxiv_{scanned}"),
                    "title": title,
                    "abstract": abstract,
                    "authors": str(rec.get("authors") or ""),
                    "categories": " ".join(cat_list),
                    "target_categories": matched_cats,
                    "primary_category": primary_cat,
                    "num_categories": len(cat_list),
                    "year": _extract_year(rec),
                    "doi": str(rec.get("doi") or ""),
                    "journal_ref": str(rec.get("journal-ref") or ""),
                }
            )

            if max_records is not None and len(rows) >= max_records:
                break

    logger.info(
        "Streamed %d lines -> collected %d ArXiv records (malformed lines skipped: %d).",
        scanned,
        len(rows),
        malformed,
    )
    return pd.DataFrame(rows)


def fetch_arxiv_api_sample(
    output_path: Path,
    categories: list[str],
    per_category: int = 50,
    delay_seconds: float = 1.0,
    timeout: int = 30,
) -> Path:
    """Fetch a manageable, real ArXiv metadata sample via the official public ArXiv Atom API.

    Writes records in Kaggle-compatible JSONL schema so downstream loaders, EDA, and
    preprocessing work identically with either the API sample or the manual Kaggle snapshot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    records: list[dict] = []

    logger.info(
        "Fetching manageable ArXiv API sample (%d categories x %d papers)...",
        len(categories),
        per_category,
    )

    for idx, cat in enumerate(categories):
        query = urllib.parse.urlencode(
            {
                "search_query": f"cat:{cat}",
                "start": 0,
                "max_results": per_category,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"{ARXIV_API_ENDPOINT}?{query}"
        resp = requests.get(
            url,
            headers={"User-Agent": "ScholarMindAI/0.1 (research-assistant-eda)"},
            timeout=timeout,
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", ATOM_NS):
            id_url = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            paper_id = id_url.rsplit("/", 1)[-1].split("v")[0] if id_url else ""
            if not paper_id or paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
            authors = ", ".join(
                (a.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
                for a in entry.findall("atom:author", ATOM_NS)
            )
            cats = [
                c.attrib.get("term", "").strip()
                for c in entry.findall("atom:category", ATOM_NS)
                if c.attrib.get("term")
            ]
            if cat not in cats:
                cats.insert(0, cat)

            records.append(
                {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": " ".join(cats),
                    "published": published,
                    "update_date": published[:10],
                    "versions": [{"version": "v1", "created": published[:4]}],
                    "doi": "",
                    "journal-ref": "",
                }
            )

        if idx < len(categories) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Saved %d ArXiv API sample records -> %s", len(records), output_path)
    return output_path
