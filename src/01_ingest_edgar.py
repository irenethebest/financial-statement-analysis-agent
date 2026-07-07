# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest — SEC EDGAR → Bronze
# MAGIC
# MAGIC Pulls XBRL **company facts** and filing metadata for a list of tickers
# MAGIC from SEC EDGAR's free JSON APIs (no scraping, no API key) and lands them
# MAGIC as managed Delta tables in Unity Catalog.
# MAGIC
# MAGIC **Tables created** in `{catalog}.bronze`:
# MAGIC * `companies`   — one row per company (identity + fetch metadata)
# MAGIC * `xbrl_facts`  — one row per XBRL observation (fully exploded)
# MAGIC * `filings`     — recent filing index (form, accession, dates)

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")
dbutils.widgets.text("tickers", "AAPL,MSFT,NVDA,TSLA", "Tickers (comma-separated)")
dbutils.widgets.text("user_agent_email", "irenejinheechoi@gmail.com",
                     "Contact email for SEC User-Agent header")
dbutils.widgets.text("owner_user", "jchoi867@gatech.edu",
                     "User to grant catalog access (CI runs as a service principal)")
dbutils.widgets.text("include_existing", "true",
                     "Also re-ingest tickers already in bronze (merge semantics)")

CATALOG = dbutils.widgets.get("catalog")
OWNER_USER = dbutils.widgets.get("owner_user")
INCLUDE_EXISTING = dbutils.widgets.get("include_existing").lower() == "true"
SCHEMA = "bronze"
TICKERS = [t.strip().upper() for t in dbutils.widgets.get("tickers").split(",") if t.strip()]
UA_EMAIL = dbutils.widgets.get("user_agent_email")
print(f"Target: {CATALOG}.{SCHEMA} | tickers: {TICKERS}")

# COMMAND ----------

import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsa import edgar  # noqa: E402

# COMMAND ----------

# Create the catalog and ALL medallion schemas up front (01 runs first).
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
for s in ("bronze", "silver", "gold"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{s}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# CI deploys run this job as a service principal, which then owns the
# catalog. Grant the human owner full access so it shows up in Catalog
# Explorer (no-op if the job runs as that user already).
if OWNER_USER:
    try:
        spark.sql(f"GRANT ALL PRIVILEGES ON CATALOG {CATALOG} "
                  f"TO `{OWNER_USER}`")
        # ALL PRIVILEGES deliberately excludes MANAGE; grant it too so the
        # owner can hand out access (e.g. to the app's service principal).
        spark.sql(f"GRANT MANAGE ON CATALOG {CATALOG} TO `{OWNER_USER}`")
        print(f"Granted ALL PRIVILEGES + MANAGE on {CATALOG} to {OWNER_USER}")
    except Exception as exc:
        print(f"Grant skipped: {exc}")

# COMMAND ----------

# MAGIC %md ### Fetch from EDGAR (rate-limited, ~7 req/s max)

# COMMAND ----------

import pandas as pd

# Merge semantics: the widget lists tickers to ADD; union with what's
# already ingested so a full overwrite refreshes everyone. This is what
# lets the agent's ingest_company tool add one ticker without dropping
# the rest.
if INCLUDE_EXISTING:
    try:
        existing = [r.ticker for r in
                    spark.table("companies").select("ticker").collect()]
        TICKERS = sorted(set(TICKERS) | set(existing))
        print(f"Union with {len(existing)} existing -> {TICKERS}")
    except Exception:
        print("No existing companies table — first run")

fetched_at = datetime.now(timezone.utc).isoformat()
companies, fact_rows, filing_rows = [], [], []

failed = []
for ticker in TICKERS:
    print(f"Fetching {ticker} ...")
    try:
        bundle = edgar.fetch_all(ticker, UA_EMAIL)
    except Exception as exc:
        # One bad ticker must not sink the whole (merged) run.
        print(f"  SKIPPED {ticker}: {exc}")
        failed.append(ticker)
        continue
    facts = edgar.facts_to_records(ticker, bundle["company_facts"])
    fact_rows.extend(facts)

    recent = bundle["submissions"].get("filings", {}).get("recent", {})
    n = len(recent.get("accessionNumber", []))
    for i in range(n):
        if recent["form"][i] in ("10-K", "10-Q", "8-K", "DEF 14A"):
            filing_rows.append({
                "ticker": ticker,
                "cik": bundle["cik"],
                "form": recent["form"][i],
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i] or None,
                "primary_document": recent["primaryDocument"][i],
            })

    companies.append({
        "ticker": ticker,
        "cik": bundle["cik"],
        "entity_name": bundle["entity_name"],
        "n_fact_rows": len(facts),
        "fetched_at": fetched_at,
    })
    print(f"  {bundle['entity_name']} (CIK {bundle['cik']}): "
          f"{len(facts):,} fact rows")

# COMMAND ----------

# MAGIC %md ### Write bronze Delta tables (idempotent overwrite — tiny data)

# COMMAND ----------

spark.createDataFrame(pd.DataFrame(companies)) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("companies")

facts_pdf = pd.DataFrame(fact_rows)
# Explicit types keep the Delta schema stable across runs.
facts_pdf["val"] = pd.to_numeric(facts_pdf["val"], errors="coerce")
facts_pdf["fy"] = pd.to_numeric(facts_pdf["fy"], errors="coerce").astype("Int64")
spark.createDataFrame(facts_pdf) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("xbrl_facts")

spark.createDataFrame(pd.DataFrame(filing_rows)) \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("filings")

print("Bronze tables written:")
for t in ("companies", "xbrl_facts", "filings"):
    print(f"  {CATALOG}.{SCHEMA}.{t}: {spark.table(t).count():,} rows")
if failed:
    print(f"Skipped tickers (not found / fetch error): {failed}")
