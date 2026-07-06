"""SEC EDGAR client — no HTML scraping, JSON APIs only.

Endpoints used (all free, no API key):
  * https://www.sec.gov/files/company_tickers.json          ticker -> CIK
  * https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json  all XBRL facts
  * https://data.sec.gov/submissions/CIK{cik}.json            filing metadata

SEC fair-access rules: declare a User-Agent with contact info, stay under
10 requests/second. https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_MIN_INTERVAL_S = 0.15  # ~6.7 req/s, under SEC's 10 req/s cap
_last_request_ts = 0.0


def _headers(user_agent_email: str) -> dict[str, str]:
    return {
        "User-Agent": f"FinanceData_Portfolio research agent {user_agent_email}",
        "Accept-Encoding": "gzip, deflate",
    }


def _get(url: str, user_agent_email: str, retries: int = 3) -> dict[str, Any]:
    """GET with rate limiting and simple exponential backoff."""
    global _last_request_ts
    for attempt in range(retries):
        wait = _MIN_INTERVAL_S - (time.time() - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.time()
        resp = requests.get(url, headers=_headers(user_agent_email), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (403, 429, 503) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url}")


def ticker_to_cik(ticker: str, user_agent_email: str) -> tuple[int, str]:
    """Resolve a ticker to (CIK, official company title)."""
    data = _get(TICKERS_URL, user_agent_email)
    ticker = ticker.upper().strip()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return int(entry["cik_str"]), entry["title"]
    raise KeyError(f"Ticker {ticker!r} not found in SEC company list")


def fetch_company_facts(cik: int, user_agent_email: str) -> dict[str, Any]:
    """All XBRL facts ever reported by the company (income stmt, balance
    sheet, cash flow line items across every filing)."""
    return _get(FACTS_URL.format(cik=cik), user_agent_email)


def fetch_submissions(cik: int, user_agent_email: str) -> dict[str, Any]:
    """Filing index — form types, accession numbers, filing dates."""
    return _get(SUBMISSIONS_URL.format(cik=cik), user_agent_email)


def fetch_all(ticker: str, user_agent_email: str) -> dict[str, Any]:
    """Convenience bundle for one company: identity + facts + submissions."""
    cik, title = ticker_to_cik(ticker, user_agent_email)
    return {
        "ticker": ticker.upper().strip(),
        "cik": cik,
        "entity_name": title,
        "company_facts": fetch_company_facts(cik, user_agent_email),
        "submissions": fetch_submissions(cik, user_agent_email),
    }


def facts_to_records(ticker: str, company_facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested companyfacts JSON into tidy rows.

    One row per (taxonomy, tag, unit, period, filing) observation:
    {ticker, taxonomy, tag, label, unit, start, end, val, fy, fp, form, filed, frame, accn}
    """
    rows: list[dict[str, Any]] = []
    for taxonomy, tags in company_facts.get("facts", {}).items():
        for tag, meta in tags.items():
            label = meta.get("label")
            for unit, observations in meta.get("units", {}).items():
                for obs in observations:
                    rows.append(
                        {
                            "ticker": ticker,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "label": label,
                            "unit": unit,
                            "start": obs.get("start"),
                            "end": obs.get("end"),
                            "val": obs.get("val"),
                            "fy": obs.get("fy"),
                            "fp": obs.get("fp"),
                            "form": obs.get("form"),
                            "filed": obs.get("filed"),
                            "frame": obs.get("frame"),
                            "accn": obs.get("accn"),
                        }
                    )
    return rows


def to_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))
