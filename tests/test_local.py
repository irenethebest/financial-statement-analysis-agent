"""Offline tests — synthetic XBRL facts through the full analysis chain.

No network, no Spark, no Databricks:  python tests/test_local.py
Builds a fake company ("TEST") with deliberately planted problems:
  * FY2025 receivables grow +50% on +10% revenue  -> ar_outpaces_revenue
  * FY2025 OCF is 40% of net income               -> earnings_not_backed_by_cash
and verifies extraction, the DuPont identity, and the anomaly rules.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fsa import anomalies, ratios, statements  # noqa: E402
from fsa.agent_core import TOOL_SPECS, build_tool_sql  # noqa: E402

FY = {  # fiscal years and planted values (USD millions for readability)
    #        rev   cogs   opinc  ni    assets  cur_a  cur_l  eq     ar    inv   ocf
    "2023": [1000, 600,   200,   150,  2000,   800,   500,   900,   100,  120,  180],
    "2024": [1100, 660,   220,   165,  2200,   880,   550,   990,   110,  132,  200],
    "2025": [1210, 726,   242,   181,  2420,   968,   605,   1089,  165,  145,   72],
}
M = 1_000_000


def synth_facts() -> pd.DataFrame:
    rows = []

    def add(tag, kind, year, val):
        end = f"{year}-12-31"
        rows.append({
            "ticker": "TEST", "taxonomy": "us-gaap", "tag": tag,
            "label": tag, "unit": "USD",
            "start": f"{int(year)-1}-12-31" if kind == "duration" else None,
            "end": end, "val": float(val) * M, "fy": int(year), "fp": "FY",
            "form": "10-K", "filed": f"{int(year)+1}-02-15",
            "frame": None, "accn": f"acc-{year}",
        })

    for year, (rev, cogs, opinc, ni, assets, cur_a, cur_l, eq, ar, inv,
               ocf) in FY.items():
        add("Revenues", "duration", year, rev)  # 2nd fallback tag on purpose
        add("CostOfGoodsAndServicesSold", "duration", year, cogs)
        add("OperatingIncomeLoss", "duration", year, opinc)
        add("NetIncomeLoss", "duration", year, ni)
        add("Assets", "instant", year, assets)
        add("AssetsCurrent", "instant", year, cur_a)
        add("LiabilitiesCurrent", "instant", year, cur_l)
        add("StockholdersEquity", "instant", year, eq)
        add("AccountsReceivableNetCurrent", "instant", year, ar)
        add("InventoryNet", "instant", year, inv)
        add("NetCashProvidedByUsedInOperatingActivities", "duration", year, ocf)
        add("CashAndCashEquivalentsAtCarryingValue", "instant", year, 300)

    # Restatement: 2023 revenue re-filed in 2025 with a corrected value —
    # the extractor must keep the most recently filed number.
    rows.append({
        "ticker": "TEST", "taxonomy": "us-gaap", "tag": "Revenues",
        "label": "Revenues", "unit": "USD", "start": "2022-12-31",
        "end": "2023-12-31", "val": 1005.0 * M, "fy": 2025, "fp": "FY",
        "form": "10-K", "filed": "2026-02-15", "frame": None,
        "accn": "acc-restated",
    })
    return pd.DataFrame(rows)


def main() -> None:
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        assert cond, f"FAIL: {msg}"
        checks += 1
        print(f"  ok — {msg}")

    print("1. Statement extraction")
    tidy = statements.build_statements(synth_facts(), form="10-K", periods=6)
    ok(not tidy.empty, "extracted rows")
    rev = tidy[tidy["line_item"] == "revenue"].set_index("period_end")["value"]
    ok(len(rev) == 3, "3 fiscal years of revenue")
    ok(rev["2023-12-31"] == 1005 * M, "restated value wins (latest filed)")
    ok((tidy[tidy["line_item"] == "revenue"]["tag_used"] == "Revenues").all(),
       "fallback tag selection")

    print("2. Wide table + derived items")
    wide = statements.to_wide(tidy)
    ok(len(wide) == 3, "3 company-periods")
    gp = wide.set_index("period_end").loc["2024-12-31", "gross_profit"]
    ok(gp == (1100 - 660) * M, "gross_profit derived from rev - cogs")

    print("3. Ratios & DuPont")
    r = ratios.compute_ratios(wide)
    row25 = r[r["period_end"] == "2025-12-31"].iloc[0]
    dupont = (row25["dupont_net_margin"] * row25["dupont_asset_turnover"]
              * row25["dupont_equity_multiplier"])
    ok(abs(dupont - row25["roe"]) < 1e-12, "DuPont identity == ROE")
    ok(abs(row25["current_ratio"] - 968 / 605) < 1e-9, "current ratio")
    ok(abs(row25["revenue_growth"] - (1210 / 1100 - 1)) < 1e-9,
       "revenue growth")

    print("4. Anomaly rules")
    flags = anomalies.detect_anomalies(anomalies.with_prior_margin(r))
    fired = set(flags[flags["period_end"] == "2025-12-31"]["rule_id"])
    ok("ar_outpaces_revenue" in fired, "planted AR>revenue flag fired")
    ok("earnings_not_backed_by_cash" in fired, "planted OCF<NI flag fired")
    ok("negative_equity" not in fired, "no false negative-equity flag")
    for _, f in flags.iterrows():
        json.loads(f["metrics"])  # metrics payloads are valid JSON
    ok(True, "flag metrics are valid JSON")

    print("5. Agent tool plumbing")
    sql = build_tool_sql("fsa.dev", "get_statements",
                         {"p_ticker": "a'; DROP TABLE x;--", "p_statement": "IS"})
    ok("DROP TABLE" in sql and "''" in sql, "quote escaping on tool args")
    names = {t["function"]["name"] for t in TOOL_SPECS}
    ok(names == {"list_companies", "get_statements", "get_ratios",
                 "get_anomalies", "ingest_company", "get_pipeline_status"},
       "tool specs complete")
    from fsa.agent_core import ACTION_TOOLS, _PARAM_ORDER
    ok(set(_PARAM_ORDER) | ACTION_TOOLS == names,
       "every tool is either SQL-backed or an action tool")

    print(f"\nAll {checks} checks passed.")


if __name__ == "__main__":
    main()
