# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — Ratios, DuPont Decomposition & Red Flags
# MAGIC
# MAGIC Computes liquidity / leverage / profitability / efficiency ratios and
# MAGIC the 3-factor DuPont ROE decomposition, then runs the rule-based
# MAGIC earnings-quality anomaly detector.
# MAGIC
# MAGIC **Tables created** in `{catalog}.gold`:
# MAGIC * `ratios`    — one row per (ticker, period), ~30 ratio columns
# MAGIC * `anomalies` — one row per fired red flag, with explanation

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")

CATALOG = dbutils.widgets.get("catalog")

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA gold")

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsa import anomalies, ratios  # noqa: E402

# COMMAND ----------

wide = spark.table(f"{CATALOG}.silver.statements_wide").toPandas()
r = ratios.compute_ratios(wide)
print(f"Computed {r.shape[1] - 2} ratios for {len(r)} company-periods")

# Sanity check: DuPont components must multiply back to ROE.
check = (r["dupont_roe_check"] - r["roe"]).abs().max()
assert check < 1e-6 or not (check == check), f"DuPont mismatch: {check}"
print("DuPont decomposition check passed")

# COMMAND ----------

flags = anomalies.detect_anomalies(anomalies.with_prior_margin(r))
print(f"{len(flags)} red flags fired")
if not flags.empty:
    display(spark.createDataFrame(
        flags[["ticker", "period_end", "rule_id", "severity"]]))

# COMMAND ----------

spark.createDataFrame(r.drop(columns=["_prev_gross_margin"], errors="ignore")) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("ratios")

if flags.empty:
    # Keep a stable (empty) schema so downstream tools never break.
    spark.sql("""
        CREATE OR REPLACE TABLE anomalies (
            ticker STRING, period_end STRING, rule_id STRING,
            severity STRING, explanation STRING, metrics STRING)
    """)
else:
    spark.createDataFrame(flags) \
        .write.mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable("anomalies")

print(f"Wrote {CATALOG}.gold.ratios ({len(r)}) and gold.anomalies ({len(flags)})")
