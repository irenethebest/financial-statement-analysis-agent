"""Extract the three core statements from flattened XBRL facts.

The hard problem this module solves: companies report the same economic
line item under different US-GAAP tags (e.g. revenue may be `Revenues`,
`RevenueFromContractWithCustomerExcludingAssessedTax`, or `SalesRevenueNet`).
Each line item below carries an ordered fallback list; the first tag with
usable data wins.
"""

from __future__ import annotations

import pandas as pd

# statement: IS = income statement, BS = balance sheet, CF = cash flow
# kind: "duration" (flow over a period) or "instant" (point-in-time balance)
LINE_ITEMS: dict[str, dict] = {
    # ---- Income statement ----
    "revenue": {
        "statement": "IS", "kind": "duration",
        "tags": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
    },
    "cost_of_revenue": {
        "statement": "IS", "kind": "duration",
        "tags": [
            "CostOfGoodsAndServicesSold",
            "CostOfRevenue",
            "CostOfGoodsSold",
            "CostOfServices",
        ],
    },
    "gross_profit": {
        "statement": "IS", "kind": "duration",
        "tags": ["GrossProfit"],
    },
    "operating_income": {
        "statement": "IS", "kind": "duration",
        "tags": ["OperatingIncomeLoss"],
    },
    "interest_expense": {
        "statement": "IS", "kind": "duration",
        "tags": ["InterestExpense", "InterestExpenseNonoperating",
                 "InterestAndDebtExpense", "InterestExpenseDebt"],
    },
    "pretax_income": {
        "statement": "IS", "kind": "duration",
        "tags": [
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ],
    },
    "income_tax_expense": {
        "statement": "IS", "kind": "duration",
        "tags": ["IncomeTaxExpenseBenefit"],
    },
    "net_income": {
        "statement": "IS", "kind": "duration",
        "tags": ["NetIncomeLoss", "ProfitLoss",
                 "NetIncomeLossAvailableToCommonStockholdersBasic"],
    },
    # ---- Balance sheet ----
    "cash_and_equivalents": {
        "statement": "BS", "kind": "instant",
        "tags": ["CashAndCashEquivalentsAtCarryingValue",
                 "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    },
    "accounts_receivable": {
        "statement": "BS", "kind": "instant",
        "tags": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                 "AccountsNotesAndLoansReceivableNetCurrent"],
    },
    "inventory": {
        "statement": "BS", "kind": "instant",
        "tags": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"],
    },
    "current_assets": {
        "statement": "BS", "kind": "instant",
        "tags": ["AssetsCurrent"],
    },
    "total_assets": {
        "statement": "BS", "kind": "instant",
        "tags": ["Assets"],
    },
    "accounts_payable": {
        "statement": "BS", "kind": "instant",
        "tags": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    },
    "current_liabilities": {
        "statement": "BS", "kind": "instant",
        "tags": ["LiabilitiesCurrent"],
    },
    "short_term_debt": {
        "statement": "BS", "kind": "instant",
        "tags": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings",
                 "CommercialPaper"],
    },
    "long_term_debt": {
        "statement": "BS", "kind": "instant",
        "tags": ["LongTermDebtNoncurrent", "LongTermDebt",
                 "LongTermDebtAndCapitalLeaseObligations"],
    },
    "total_liabilities": {
        "statement": "BS", "kind": "instant",
        "tags": ["Liabilities"],
    },
    "stockholders_equity": {
        "statement": "BS", "kind": "instant",
        "tags": ["StockholdersEquity",
                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    },
    # ---- Cash flow ----
    "operating_cash_flow": {
        "statement": "CF", "kind": "duration",
        "tags": ["NetCashProvidedByUsedInOperatingActivities",
                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    },
    "capex": {
        "statement": "CF", "kind": "duration",
        "tags": ["PaymentsToAcquirePropertyPlantAndEquipment",
                 "PaymentsToAcquireProductiveAssets"],
    },
    "investing_cash_flow": {
        "statement": "CF", "kind": "duration",
        "tags": ["NetCashProvidedByUsedInInvestingActivities",
                 "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    },
    "financing_cash_flow": {
        "statement": "CF", "kind": "duration",
        "tags": ["NetCashProvidedByUsedInFinancingActivities",
                 "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    },
    "depreciation_amortization": {
        "statement": "CF", "kind": "duration",
        "tags": ["DepreciationDepletionAndAmortization",
                 "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    },
}

_ANNUAL_MIN_DAYS, _ANNUAL_MAX_DAYS = 340, 390
_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS = 80, 100


def _select_periods(df: pd.DataFrame, kind: str, form: str) -> pd.DataFrame:
    """Filter one tag's observations to clean fiscal periods for `form`."""
    df = df[df["form"] == form].copy()
    if df.empty:
        return df
    df["end_dt"] = pd.to_datetime(df["end"])
    if kind == "duration":
        df = df.dropna(subset=["start"])
        df["start_dt"] = pd.to_datetime(df["start"])
        days = (df["end_dt"] - df["start_dt"]).dt.days
        if form == "10-K":
            df = df[days.between(_ANNUAL_MIN_DAYS, _ANNUAL_MAX_DAYS)]
        else:  # 10-Q: keep discrete quarters only (not cumulative YTD)
            df = df[days.between(_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS)]
    # Same period can appear in multiple filings (restatements, comparatives).
    # Keep the most recently filed value per period end.
    df = df.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
    return df


def build_statements(
    facts: pd.DataFrame,
    form: str = "10-K",
    periods: int = 6,
) -> pd.DataFrame:
    """Tidy statement table from flattened facts.

    Returns one row per (ticker, line_item, period_end):
    [ticker, statement, line_item, period_end, fiscal_year, fiscal_period,
     form, value, tag_used, filed]
    """
    usd = facts[(facts["taxonomy"] == "us-gaap") & (facts["unit"] == "USD")]
    out_frames: list[pd.DataFrame] = []

    for ticker, tfacts in usd.groupby("ticker"):
        for item, spec in LINE_ITEMS.items():
            picked = None
            for tag in spec["tags"]:
                cand = _select_periods(
                    tfacts[tfacts["tag"] == tag], spec["kind"], form
                )
                # Require decent coverage before accepting a tag; otherwise
                # try the next fallback.
                if len(cand) >= min(2, periods):
                    picked = cand.assign(tag_used=tag)
                    break
                if picked is None and not cand.empty:
                    picked = cand.assign(tag_used=tag)
            if picked is None or picked.empty:
                continue
            picked = picked.sort_values("end_dt").tail(periods)
            out_frames.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "statement": spec["statement"],
                        "line_item": item,
                        "period_end": picked["end"].values,
                        "fiscal_year": picked["fy"].values,
                        "fiscal_period": picked["fp"].values,
                        "form": form,
                        "value": picked["val"].astype("float64").values,
                        "tag_used": picked["tag_used"].values,
                        "filed": picked["filed"].values,
                    }
                )
            )

    if not out_frames:
        return pd.DataFrame(
            columns=["ticker", "statement", "line_item", "period_end",
                     "fiscal_year", "fiscal_period", "form", "value",
                     "tag_used", "filed"]
        )
    return pd.concat(out_frames, ignore_index=True)


def to_wide(statements: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy statements to one row per (ticker, period_end) with a
    column per line item. Derives items that are often unreported:
    gross_profit and total_liabilities."""
    wide = (
        statements.pivot_table(
            index=["ticker", "period_end"],
            columns="line_item",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in LINE_ITEMS:
        if col not in wide.columns:
            wide[col] = float("nan")
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")

    wide["gross_profit"] = wide["gross_profit"].fillna(
        wide["revenue"] - wide["cost_of_revenue"]
    )
    wide["total_liabilities"] = wide["total_liabilities"].fillna(
        wide["total_assets"] - wide["stockholders_equity"]
    )
    wide["total_debt"] = wide[["short_term_debt", "long_term_debt"]].sum(
        axis=1, min_count=1
    )
    return wide.sort_values(["ticker", "period_end"]).reset_index(drop=True)
