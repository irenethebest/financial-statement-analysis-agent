# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Register Agent Tools as Unity Catalog Functions
# MAGIC
# MAGIC Each tool is a **SQL UDF in Unity Catalog** — governed, discoverable,
# MAGIC and callable by any agent (or any user with EXECUTE). The functions
# MAGIC return JSON strings so the LLM can read them directly.
# MAGIC
# MAGIC Functions live in `{catalog}.gold` (the serving layer) and read across
# MAGIC the medallion schemas: `bronze.companies`, `silver.statements`,
# MAGIC `gold.ratios`, `gold.anomalies`.
# MAGIC
# MAGIC The `COMMENT` on each function/parameter matters: the agent framework
# MAGIC turns them into the tool descriptions the LLM sees.

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")

CATALOG = dbutils.widgets.get("catalog")
FQ = f"{CATALOG}.gold"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA gold")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.list_companies()
RETURNS STRING
COMMENT 'Lists the companies available for analysis. Returns a JSON array of objects with ticker, cik, and entity_name. Call this first if unsure which tickers exist.'
RETURN (
  SELECT to_json(collect_list(struct(ticker, cik, entity_name)))
  FROM {CATALOG}.bronze.companies
)
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.get_statements(
  p_ticker STRING COMMENT 'Stock ticker, e.g. AAPL',
  p_statement STRING COMMENT 'Which statement: IS (income statement), BS (balance sheet), or CF (cash flow)'
)
RETURNS STRING
COMMENT 'Returns the extracted financial statement line items for a company across recent fiscal periods, as a JSON array of {{line_item, period_end, value, tag_used}}. Values are USD as reported in SEC filings.'
RETURN (
  SELECT to_json(collect_list(struct(line_item, period_end, value, tag_used)))
  FROM {CATALOG}.silver.statements
  WHERE ticker = upper(p_ticker) AND statement = upper(p_statement)
)
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.get_ratios(
  p_ticker STRING COMMENT 'Stock ticker, e.g. AAPL'
)
RETURNS STRING
COMMENT 'Returns all computed financial ratios for a company across recent fiscal years as a JSON array: liquidity (current/quick/cash ratio), leverage (debt_to_equity, interest_coverage), profitability (margins, ROA, ROE), the 3-factor DuPont ROE decomposition (dupont_net_margin x dupont_asset_turnover x dupont_equity_multiplier), efficiency (DSO/DIO/DPO days, cash conversion cycle), cash quality (ocf_to_net_income, free_cash_flow, accruals_ratio), and YoY growth rates (fractions: 0.12 = +12%). All ratios are precomputed deterministically — report them as-is, never recalculate.'
RETURN (
  SELECT to_json(collect_list(struct(*)))
  FROM {CATALOG}.gold.ratios
  WHERE ticker = upper(p_ticker)
)
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.get_anomalies(
  p_ticker STRING COMMENT 'Stock ticker, e.g. AAPL'
)
RETURNS STRING
COMMENT 'Returns rule-based earnings-quality red flags for a company as a JSON array of {{period_end, rule_id, severity, explanation, metrics}}. Severity is info/medium/high. These are the ONLY anomalies to report — do not invent additional flags. An empty array means no rules fired.'
RETURN (
  SELECT to_json(collect_list(struct(period_end, rule_id, severity, explanation, metrics)))
  FROM {CATALOG}.gold.anomalies
  WHERE ticker = upper(p_ticker)
)
""")

# COMMAND ----------

# MAGIC %md ### Smoke test

# COMMAND ----------

for fn, args in [("list_companies", ""),
                 ("get_statements", "'AAPL','IS'"),
                 ("get_ratios", "'AAPL'"),
                 ("get_anomalies", "'AAPL'")]:
    out = spark.sql(f"SELECT {FQ}.{fn}({args}) AS r").first()["r"]
    preview = (out or "NULL")[:160]
    print(f"{fn}({args}) -> {preview} ...")
