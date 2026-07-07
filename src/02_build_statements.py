# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — Bronze XBRL Facts → Clean Financial Statements
# MAGIC
# MAGIC Maps raw US-GAAP tags to canonical line items (with ordered fallback
# MAGIC lists per item — companies tag the same concept differently), selects
# MAGIC clean fiscal periods, and dedupes restated values by latest filing.
# MAGIC
# MAGIC **Tables created** in `{catalog}.silver`:
# MAGIC * `statements`      — tidy: one row per (ticker, line_item, period)
# MAGIC * `statements_wide` — one row per (ticker, period), column per item
# MAGIC * `mdna_chunks`     — MD&A split into ~1.4k-char paragraph chunks
# MAGIC   for the agent's keyword search tool

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")
dbutils.widgets.text("form", "10-K", "Filing form (10-K annual / 10-Q quarterly)")
dbutils.widgets.text("periods", "5", "Annual periods to keep (fiscal years)")
dbutils.widgets.text("q_periods", "8", "Quarterly periods to keep (10-Q)")

CATALOG = dbutils.widgets.get("catalog")
FORM = dbutils.widgets.get("form")
PERIODS = int(dbutils.widgets.get("periods"))
Q_PERIODS = int(dbutils.widgets.get("q_periods"))

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA silver")

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsa import mdna, statements  # noqa: E402

# COMMAND ----------

facts = spark.table(f"{CATALOG}.bronze.xbrl_facts").toPandas()
print(f"Loaded {len(facts):,} fact rows")

tidy = statements.build_statements(facts, form=FORM, periods=PERIODS)
wide = statements.to_wide(tidy)
print(f"Statements: {len(tidy):,} tidy rows -> {len(wide):,} company-periods")

# COMMAND ----------

# Which XBRL tag was used per line item — provenance for auditability.
coverage = (
    tidy.groupby(["ticker", "line_item", "tag_used"])
    .size().reset_index(name="n_periods")
)
display(spark.createDataFrame(coverage))

# COMMAND ----------

spark.createDataFrame(tidy) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("statements")

spark.createDataFrame(wide) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("statements_wide")

print(f"Wrote {CATALOG}.silver.statements ({len(tidy):,}) and "
      f"statements_wide ({len(wide):,})")

# COMMAND ----------

# MAGIC %md ### Quarterly (10-Q) statements — discrete quarters only
# MAGIC The extractor filters to ~90-day durations, so cumulative
# MAGIC year-to-date figures reported in 10-Qs are excluded.

# COMMAND ----------

tidy_q = statements.build_statements(facts, form="10-Q", periods=Q_PERIODS)
wide_q = statements.to_wide(tidy_q)
print(f"Quarterly: {len(tidy_q):,} tidy rows -> {len(wide_q):,} "
      "company-quarters")

spark.createDataFrame(tidy_q) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("statements_q")
spark.createDataFrame(wide_q) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("statements_wide_q")
print(f"Wrote {CATALOG}.silver.statements_q and statements_wide_q")

# COMMAND ----------

# MAGIC %md ### MD&A → searchable paragraph chunks

# COMMAND ----------

chunk_rows = []
try:
    mdna_pdf = spark.table(f"{CATALOG}.bronze.mdna").toPandas()
except Exception:
    mdna_pdf = None

if mdna_pdf is not None and not mdna_pdf.empty:
    for _, row in mdna_pdf[mdna_pdf["extraction_ok"] == True].iterrows():  # noqa: E712
        for i, chunk in enumerate(mdna.chunk_text(row["mdna_text"])):
            chunk_rows.append({
                "ticker": row["ticker"],
                "form": row["form"],
                "filing_date": row["filing_date"],
                "accession_number": row["accession_number"],
                "chunk_id": i,
                "chunk_text": chunk,
            })

if chunk_rows:
    import pandas as pd
    spark.createDataFrame(pd.DataFrame(chunk_rows)) \
        .write.mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable("mdna_chunks")
else:  # stable empty schema
    spark.sql("""
        CREATE TABLE IF NOT EXISTS mdna_chunks (
            ticker STRING, form STRING, filing_date STRING,
            accession_number STRING, chunk_id INT, chunk_text STRING)
    """)
print(f"Wrote {CATALOG}.silver.mdna_chunks ({len(chunk_rows):,} chunks)")
