"""Rule-based red-flag detection (earnings quality + balance-sheet health).

Every rule is deterministic and explainable: it emits the metric values it
fired on plus a plain-English explanation. The agent narrates these — it
never invents flags of its own.
"""

from __future__ import annotations

import json

import pandas as pd

SEVERITIES = ("info", "medium", "high")


def _flag(row, rule_id, severity, explanation, metrics: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "period_end": row["period_end"],
        "rule_id": rule_id,
        "severity": severity,
        "explanation": explanation,
        "metrics": json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in metrics.items()}
        ),
    }


def detect_anomalies(ratios: pd.DataFrame) -> pd.DataFrame:
    """Input: output of compute_ratios(). Output: one row per fired flag:
    [ticker, period_end, rule_id, severity, explanation, metrics]."""
    flags: list[dict] = []

    for _, r in ratios.iterrows():
        rev_g = r.get("revenue_growth")
        ar_g = r.get("accounts_receivable_growth")
        inv_g = r.get("inventory_growth")
        cogs_g = r.get("cost_of_revenue_growth")

        # 1. Receivables outrunning revenue -> possible aggressive revenue
        #    recognition / channel stuffing.
        if pd.notna(rev_g) and pd.notna(ar_g) and ar_g > rev_g + 0.10 and ar_g > 0:
            flags.append(_flag(
                r, "ar_outpaces_revenue",
                "high" if ar_g > rev_g + 0.25 else "medium",
                "Accounts receivable grew much faster than revenue. The "
                "company is booking sales it hasn't collected cash for — a "
                "classic earnings-quality red flag (aggressive revenue "
                "recognition or customers struggling to pay).",
                {"revenue_growth": rev_g, "accounts_receivable_growth": ar_g,
                 "dso_days": r.get("dso_days")},
            ))

        # 2. Profits not backed by cash.
        ocf_ni = r.get("ocf_to_net_income")
        if pd.notna(ocf_ni) and 0 < ocf_ni < 0.8:
            flags.append(_flag(
                r, "earnings_not_backed_by_cash",
                "high" if ocf_ni < 0.5 else "medium",
                "Operating cash flow is well below net income, so reported "
                "profit relies on accruals rather than cash actually "
                "collected. Persistent gaps like this often precede "
                "write-downs or restatements.",
                {"ocf_to_net_income": ocf_ni,
                 "accruals_ratio": r.get("accruals_ratio")},
            ))

        # 3. Inventory building faster than cost of sales.
        if pd.notna(inv_g) and pd.notna(cogs_g) and inv_g > cogs_g + 0.15 and inv_g > 0:
            flags.append(_flag(
                r, "inventory_buildup", "medium",
                "Inventory grew much faster than cost of goods sold — "
                "product may be piling up unsold, risking future "
                "markdowns or write-offs.",
                {"inventory_growth": inv_g, "cost_of_revenue_growth": cogs_g,
                 "dio_days": r.get("dio_days")},
            ))

        # 4. Margin compression while revenue grows.
        gm = r.get("gross_margin")
        if pd.notna(rev_g) and rev_g > 0 and pd.notna(gm):
            prev_gm = r.get("_prev_gross_margin")
            if pd.notna(prev_gm) and gm < prev_gm - 0.03:
                flags.append(_flag(
                    r, "margin_compression", "medium",
                    "Revenue is growing but gross margin dropped sharply — "
                    "growth may be coming from discounting or lower-quality "
                    "sales.",
                    {"revenue_growth": rev_g, "gross_margin": gm,
                     "prior_gross_margin": prev_gm},
                ))

        # 5. Liquidity strain.
        cr = r.get("current_ratio")
        if pd.notna(cr) and cr < 1.0:
            flags.append(_flag(
                r, "current_ratio_below_1",
                "high" if cr < 0.8 else "medium",
                "Current liabilities exceed current assets — the company "
                "owes more in the next 12 months than it holds in "
                "short-term resources. Not always fatal (some strong "
                "businesses run negative working capital) but worth "
                "understanding why.",
                {"current_ratio": cr, "quick_ratio": r.get("quick_ratio")},
            ))

        # 6. Leverage spike.
        debt_g = r.get("total_debt_growth")
        de = r.get("debt_to_equity")
        if pd.notna(debt_g) and debt_g > 0.30 and pd.notna(de) and de > 1.0:
            flags.append(_flag(
                r, "leverage_spike", "medium",
                "Total debt jumped more than 30% year-over-year and now "
                "exceeds shareholders' equity — rising fixed obligations "
                "amplify any downturn.",
                {"total_debt_growth": debt_g, "debt_to_equity": de,
                 "interest_coverage": r.get("interest_coverage")},
            ))

        # 7. Negative equity.
        em = r.get("dupont_equity_multiplier")
        if pd.notna(em) and em < 0:
            flags.append(_flag(
                r, "negative_equity", "high",
                "Shareholders' equity is negative — cumulative losses or "
                "buybacks have wiped out the book value of the company. "
                "Standard leverage ratios stop being meaningful.",
                {"dupont_equity_multiplier": em},
            ))

        # 8. Revenue decline.
        if pd.notna(rev_g) and rev_g < -0.10:
            flags.append(_flag(
                r, "revenue_decline", "info",
                "Revenue fell more than 10% year-over-year.",
                {"revenue_growth": rev_g},
            ))

    cols = ["ticker", "period_end", "rule_id", "severity", "explanation",
            "metrics"]
    return pd.DataFrame(flags, columns=cols)


def with_prior_margin(ratios: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-period gross margin (helper for rule 4)."""
    df = ratios.sort_values(["ticker", "period_end"]).copy()
    df["_prev_gross_margin"] = df.groupby("ticker")["gross_margin"].shift(1)
    return df
