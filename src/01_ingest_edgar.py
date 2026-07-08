# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest — SEC EDGAR → Bronze
# MAGIC
# MAGIC Pulls XBRL **company facts**, filing metadata, and the **MD&A
# MAGIC narrative** for a list of tickers from SEC EDGAR (free JSON APIs +
# MAGIC the filing documents themselves) and lands managed Delta tables in
# MAGIC Unity Catalog.
# MAGIC
# MAGIC **Tables created** in `{catalog}.bronze`:
# MAGIC * `companies`   — one row per company (identity + fetch metadata)
# MAGIC * `xbrl_facts`  — one row per XBRL observation (fully exploded)
# MAGIC * `filings`     — recent filing index (form, accession, dates)
# MAGIC * `mdna`        — extracted Management's Discussion & Analysis text
# MAGIC   per recent 10-K (the raw ~15 MB filing HTML is not persisted —
# MAGIC   only the extracted section)

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")
dbutils.widgets.text("tickers", "AAPL,MSFT,NVDA,TSLA", "Tickers (comma-separated)")
dbutils.widgets.text("user_agent_email", "irenejinheechoi@gmail.com",
                     "Contact email for SEC User-Agent header")
dbutils.widgets.text("owner_user", "jchoi867@gatech.edu",
                     "User to grant catalog access (CI runs as a service principal)")
dbutils.widgets.text("include_existing", "true",
                     "Also re-ingest tickers already in bronze (merge semantics)")
dbutils.widgets.text("mdna_filings", "2",
                     "How many recent 10-Ks to extract MD&A from, per company")

CATALOG = dbutils.widgets.get("catalog")
OWNER_USER = dbutils.widgets.get("owner_user")
INCLUDE_EXISTING = dbutils.widgets.get("include_existing").lower() == "true"
MDNA_FILINGS = int(dbutils.widgets.get("mdna_filings"))
SCHEMA = "bronze"
TICKERS = [t.strip().upper() for t in dbutils.widgets.get("tickers").split(",") if t.strip()]
REQUESTED = set(TICKERS)  # explicitly asked for in THIS run
UA_EMAIL = dbutils.widgets.get("user_agent_email")
print(f"Target: {CATALOG}.{SCHEMA} | tickers: {TICKERS}")

# COMMAND ----------

import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsa import edgar, mdna, segments  # noqa: E402

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
companies, fact_rows, filing_rows, profile_rows = [], [], [], []

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

    # Company profile from the submissions JSON (sector, HQ, ...).
    sub = bundle["submissions"]
    addr = (sub.get("addresses") or {}).get("business") or {}
    profile_rows.append({
        "ticker": ticker,
        "cik": bundle["cik"],
        "entity_name": bundle["entity_name"],
        "sic": str(sub.get("sic") or ""),
        "sic_description": sub.get("sicDescription"),
        "website": sub.get("website") or None,
        "state_of_incorporation": sub.get("stateOfIncorporation"),
        "fiscal_year_end": sub.get("fiscalYearEnd"),
        "hq_street": addr.get("street1"),
        "hq_city": addr.get("city"),
        "hq_state": addr.get("stateOrCountry"),
        "hq_zip": addr.get("zipCode"),
        "phone": sub.get("phone"),
        "exchange": (sub.get("exchanges") or [None])[0],
    })
    print(f"  {bundle['entity_name']} (CIK {bundle['cik']}): "
          f"{len(facts):,} fact rows")

# COMMAND ----------

# MAGIC %md ### Extract MD&A from the most recent 10-Ks

# COMMAND ----------

mdna_rows = []
segment_rows = []
by_ticker: dict = {}
for f in filing_rows:
    if f["form"] == "10-K":
        by_ticker.setdefault(f["ticker"], []).append(f)

for ticker, fils in by_ticker.items():
    fils = sorted(fils, key=lambda f: f["filing_date"], reverse=True)
    for idx, f in enumerate(fils[:MDNA_FILINGS]):
        try:
            html = edgar.fetch_filing_document(
                f["cik"], f["accession_number"], f["primary_document"],
                UA_EMAIL)
            text = mdna.extract_mdna(mdna.html_to_text(html), form="10-K")
            if idx == 0:  # segment revenue from the latest 10-K only
                segs = segments.parse_segment_revenue(html)
                for s in segs:
                    segment_rows.append({"ticker": ticker, **s,
                                         "accession_number":
                                             f["accession_number"]})
                print(f"  Segments {ticker}: "
                      f"{[(s['label'], round(s['value']/1e9, 1)) for s in segs]}"
                      if segs else f"  Segments {ticker}: none parsed")
        except Exception as exc:
            print(f"  MD&A fetch failed for {ticker} "
                  f"{f['accession_number']}: {exc}")
            text = None
        mdna_rows.append({
            "ticker": ticker,
            "cik": f["cik"],
            "form": f["form"],
            "accession_number": f["accession_number"],
            "filing_date": f["filing_date"],
            "mdna_text": text,
            "char_count": len(text) if text else 0,
            "extraction_ok": text is not None,
        })
        print(f"  MD&A {ticker} {f['filing_date']}: "
              f"{len(text):,} chars" if text else
              f"  MD&A {ticker} {f['filing_date']}: NOT FOUND")

# COMMAND ----------

# MAGIC %md ### Write bronze Delta tables (idempotent overwrite — tiny data)

# COMMAND ----------

fetched = [c["ticker"] for c in companies]


def write_merge(table: str, sdf):
    """Per-ticker merge: replace only THIS run's tickers, keep the rest.
    Existing companies are no longer re-fetched on incremental runs."""
    if spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{table}"):
        tick_list = ",".join(f"'{t}'" for t in fetched)
        spark.sql(f"DELETE FROM {table} WHERE ticker IN ({tick_list})")
        sdf.write.mode("append").option("mergeSchema", "true") \
            .saveAsTable(table)
    else:
        sdf.write.mode("overwrite").option("overwriteSchema", "true") \
            .saveAsTable(table)


write_merge("companies", spark.createDataFrame(pd.DataFrame(companies)))

facts_pdf = pd.DataFrame(fact_rows)
# Explicit types keep the Delta schema stable across runs.
facts_pdf["val"] = pd.to_numeric(facts_pdf["val"], errors="coerce")
facts_pdf["fy"] = pd.to_numeric(facts_pdf["fy"], errors="coerce").astype("Int64")
write_merge("xbrl_facts", spark.createDataFrame(facts_pdf))

write_merge("filings", spark.createDataFrame(pd.DataFrame(filing_rows)))

# astype("string"): all-None columns (e.g. website) otherwise break
# Spark's type inference and fail the write.
profile_pdf = pd.DataFrame(profile_rows)
for c in profile_pdf.columns:
    if c != "cik":
        profile_pdf[c] = profile_pdf[c].astype("string")
write_merge("company_profile", spark.createDataFrame(profile_pdf))

if mdna_rows:
    write_merge("mdna", spark.createDataFrame(pd.DataFrame(mdna_rows)))
else:  # stable empty schema so downstream never breaks
    spark.sql("""
        CREATE TABLE IF NOT EXISTS mdna (
            ticker STRING, cik BIGINT, form STRING,
            accession_number STRING, filing_date STRING,
            mdna_text STRING, char_count BIGINT, extraction_ok BOOLEAN)
    """)

if segment_rows:
    write_merge("segment_revenue",
                spark.createDataFrame(pd.DataFrame(segment_rows)))
else:
    spark.sql("""
        CREATE TABLE IF NOT EXISTS segment_revenue (
            ticker STRING, axis STRING, member STRING, label STRING,
            value DOUBLE, `start` STRING, `end` STRING,
            accession_number STRING)
    """)

print("Bronze tables written:")
for t in ("companies", "xbrl_facts", "filings", "mdna", "segment_revenue"):
    print(f"  {CATALOG}.{SCHEMA}.{t}: {spark.table(t).count():,} rows")
if failed:
    print(f"Skipped tickers (not found / fetch error): {failed}")
    # A pre-existing ticker failing a refresh is tolerable; the ticker this
    # run was EXPLICITLY asked to ingest failing is not — fail loudly so
    # the agent's get_pipeline_status reports it instead of silently
    # re-triggering forever.
    bad = sorted(set(failed) & REQUESTED)
    if bad:
        raise RuntimeError(
            f"Requested ticker(s) failed to ingest: {bad} — see the "
            "SKIPPED lines above for the underlying error.")
