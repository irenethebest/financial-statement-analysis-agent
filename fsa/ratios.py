"""Ratio computation on the wide statement table.

Groups: liquidity, leverage, profitability, efficiency, cash quality,
plus the 3-factor DuPont ROE decomposition
(ROE = net margin x asset turnover x equity multiplier).
Balance-sheet denominators use period averages when a prior period exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _avg(curr: pd.Series, prev: pd.Series) -> pd.Series:
    """Average of current and prior balance; falls back to current."""
    return ((curr + prev) / 2).fillna(curr)


def compute_ratios(wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, period_end) with ratio columns.

    Expects the output of statements.to_wide(). Safe on missing inputs —
    a ratio is NaN when its ingredients are unreported.
    """
    df = wide.sort_values(["ticker", "period_end"]).copy()
    g = df.groupby("ticker")
    prev = g.shift(1)

    r = df[["ticker", "period_end"]].copy()

    # ---- Liquidity ----
    r["current_ratio"] = df["current_assets"] / df["current_liabilities"]
    r["quick_ratio"] = (
        df["current_assets"] - df["inventory"].fillna(0)
    ) / df["current_liabilities"]
    r["cash_ratio"] = df["cash_and_equivalents"] / df["current_liabilities"]

    # ---- Leverage ----
    r["debt_to_equity"] = df["total_debt"] / df["stockholders_equity"]
    r["liabilities_to_assets"] = df["total_liabilities"] / df["total_assets"]
    r["interest_coverage"] = df["operating_income"] / df["interest_expense"]

    # ---- Profitability ----
    r["gross_margin"] = df["gross_profit"] / df["revenue"]
    r["operating_margin"] = df["operating_income"] / df["revenue"]
    r["net_margin"] = df["net_income"] / df["revenue"]
    r["effective_tax_rate"] = df["income_tax_expense"] / df["pretax_income"]

    avg_assets = _avg(df["total_assets"], prev["total_assets"])
    avg_equity = _avg(df["stockholders_equity"], prev["stockholders_equity"])
    r["roa"] = df["net_income"] / avg_assets
    r["roe"] = df["net_income"] / avg_equity

    # ---- DuPont decomposition (multiplies back to ROE) ----
    r["dupont_net_margin"] = r["net_margin"]
    r["dupont_asset_turnover"] = df["revenue"] / avg_assets
    r["dupont_equity_multiplier"] = avg_assets / avg_equity
    r["dupont_roe_check"] = (
        r["dupont_net_margin"]
        * r["dupont_asset_turnover"]
        * r["dupont_equity_multiplier"]
    )

    # ---- Efficiency (365-day convention) ----
    avg_ar = _avg(df["accounts_receivable"], prev["accounts_receivable"])
    avg_inv = _avg(df["inventory"], prev["inventory"])
    avg_ap = _avg(df["accounts_payable"], prev["accounts_payable"])
    r["dso_days"] = 365 * avg_ar / df["revenue"]
    r["dio_days"] = 365 * avg_inv / df["cost_of_revenue"]
    r["dpo_days"] = 365 * avg_ap / df["cost_of_revenue"]
    r["cash_conversion_cycle_days"] = (
        r["dso_days"] + r["dio_days"] - r["dpo_days"]
    )

    # ---- Cash / earnings quality ----
    r["ocf_to_net_income"] = df["operating_cash_flow"] / df["net_income"]
    r["free_cash_flow"] = df["operating_cash_flow"] - df["capex"].fillna(0)
    r["fcf_margin"] = r["free_cash_flow"] / df["revenue"]
    r["accruals_ratio"] = (
        (df["net_income"] - df["operating_cash_flow"]) / avg_assets
    )

    # ---- YoY growth (fractions, e.g. 0.12 = +12%) ----
    for col in ["revenue", "net_income", "accounts_receivable", "inventory",
                "cost_of_revenue", "operating_cash_flow", "total_debt"]:
        r[f"{col}_growth"] = (df[col] - prev[col]) / prev[col].abs()

    return r.replace([np.inf, -np.inf], np.nan)
