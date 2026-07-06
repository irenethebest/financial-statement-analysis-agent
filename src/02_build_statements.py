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

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")
dbutils.widgets.text("form", "10-K", "Filing form (10-K annual / 10-Q quarterly)")
dbutils.widgets.text("periods", "6", "Number of periods to keep")

CATALOG = dbutils.widgets.get("catalog")
FORM = dbutils.widgets.get("form")
PERIODS = int(dbutils.widgets.get("periods"))

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA silver")

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsa import statements  # noqa: E402

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
