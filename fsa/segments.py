"""Segment revenue extraction from inline XBRL (the 10-K HTML itself).

The company-facts JSON API returns only consolidated figures — revenue by
Products/Services or by segment is *dimensional* XBRL and lives inside the
filing document as inline XBRL (`<ix:nonFraction>` facts referencing
`<xbrli:context>` definitions with explicit dimension members).

This module parses both with regex (stdlib-only, best-effort by design):
works for filers that tag disaggregated revenue on ProductOrServiceAxis or
a segment axis (Apple, Microsoft, ...). When nothing parses, callers get
an empty list and the UI degrades gracefully.
"""

from __future__ import annotations

import re

# Revenue tags to look for (mirrors statements.LINE_ITEMS["revenue"]).
REVENUE_TAGS = {
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:RevenuesNetOfInterestExpense",  # banks
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
}

# Dimension axes that represent a revenue disaggregation we can show.
_AXES = (
    "srt:ProductOrServiceAxis",
    "us-gaap:StatementBusinessSegmentsAxis",
)

_CONTEXT_RE = re.compile(
    r"<xbrli:context[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</xbrli:context>",
    re.S | re.I)
_MEMBER_RE = re.compile(
    r"<xbrldi:explicitMember[^>]*dimension=\"([^\"]+)\"[^>]*>\s*([^<\s]+)\s*<",
    re.I)
_START_RE = re.compile(r"<xbrli:startDate>([\d\-]+)</xbrli:startDate>", re.I)
_END_RE = re.compile(r"<xbrli:endDate>([\d\-]+)</xbrli:endDate>", re.I)
_FACT_RE = re.compile(
    r"<ix:nonFraction([^>]*)>(.*?)</ix:nonFraction>", re.S | re.I)
_ATTR_RE = re.compile(r"([\w:\-]+)=\"([^\"]*)\"")


def _pretty_member(member: str) -> str:
    """'aapl:IPhoneMember' -> 'IPhone', 'us-gaap:ServiceMember' -> 'Service'."""
    local = member.split(":")[-1]
    local = re.sub(r"(Segment)?Member$", "", local)
    # CamelCase -> spaced words, keeping acronym runs together.
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local).strip() or member


def parse_segment_revenue(html: str) -> list[dict]:
    """Extract disaggregated revenue for the most recent ~annual period.

    Returns [{axis, member, label, value, start, end}], one row per
    dimension member, deduped, for the latest fiscal year found.
    """
    # 1. Contexts: id -> (start, end, {axis: member}) — only single-axis
    #    contexts on an axis we care about (multi-axis = nested breakdown).
    contexts: dict[str, tuple[str, str, str, str]] = {}
    for cid, body in _CONTEXT_RE.findall(html):
        members = _MEMBER_RE.findall(body)
        axes = [(a, m) for a, m in members if a in _AXES]
        if len(members) != 1 or not axes:
            continue
        s_m, e_m = _START_RE.search(body), _END_RE.search(body)
        if not (s_m and e_m):
            continue
        contexts[cid] = (s_m.group(1), e_m.group(1), axes[0][0], axes[0][1])

    if not contexts:
        return []

    # 2. Facts: revenue tags whose contextRef is one of those contexts.
    rows: list[dict] = []
    for attrs_raw, text in _FACT_RE.findall(html):
        attrs = dict(_ATTR_RE.findall(attrs_raw))
        if attrs.get("name") not in REVENUE_TAGS:
            continue
        ctx = contexts.get(attrs.get("contextRef", ""))
        if ctx is None:
            continue
        start, end, axis, member = ctx
        raw = re.sub(r"<[^>]+>", "", text)
        raw = raw.replace(",", "").replace(" ", "").strip()
        if not re.fullmatch(r"\d+(\.\d+)?", raw):
            continue
        value = float(raw) * (10 ** int(attrs.get("scale", 0) or 0))
        if attrs.get("sign") == "-":
            value = -value
        rows.append({"axis": axis, "member": member,
                     "label": _pretty_member(member),
                     "value": value, "start": start, "end": end})

    if not rows:
        return []

    # 3. Keep the latest annual period (10-Ks restate prior years too),
    #    preferring ProductOrServiceAxis over segment axis when both exist.
    def _days(r):
        from datetime import date
        y1, m1, d1 = map(int, r["start"].split("-"))
        y2, m2, d2 = map(int, r["end"].split("-"))
        return (date(y2, m2, d2) - date(y1, m1, d1)).days

    annual = [r for r in rows if 340 <= _days(r) <= 390]
    if not annual:
        return []
    latest_end = max(r["end"] for r in annual)
    latest = [r for r in annual if r["end"] == latest_end]
    for axis in _AXES:  # first axis with data wins
        picked = [r for r in latest if r["axis"] == axis]
        if picked:
            # Dedupe members (same fact often appears in several tables).
            out: dict[str, dict] = {}
            for r in picked:
                out.setdefault(r["member"], r)
            return sorted(out.values(), key=lambda r: -r["value"])
    return []
