# Financial Statement Analysis Agent

An AI agent that reads SEC filings and explains them to non-accountants — built end-to-end on Databricks (Free Edition compatible).

Ask it about any US-listed company. If the company isn't in the data yet, the agent validates the ticker against SEC's registry and triggers the governed ingestion pipeline itself, then analyzes: five years of extracted statements, ~30 ratios including a DuPont ROE decomposition, rule-based earnings-quality red flags, segment revenue, and management's own narrative from the MD&A. The core design rule: **the LLM never computes a number** — all math is deterministic pipeline code and Unity-Catalog-governed SQL functions; the model only decides which tools to call and narrates the results.

## Architecture

```
SEC EDGAR JSON APIs + filing documents (no scraping, no key)
        │  01_ingest_edgar          rate-limited client, per-ticker MERGE writes
        ▼                           (incremental: only new tickers are fetched)
BRONZE  companies · xbrl_facts · filings · company_profile · mdna · segment_revenue
        │  02_build_statements      US-GAAP tag fallbacks (incl. bank tags),
        ▼                           annual (5y) + quarterly (8q), restatement dedupe
SILVER  statements · statements_wide · statements_q · statements_wide_q · mdna_chunks
        │  03_ratios_anomalies      liquidity/leverage/profitability/efficiency,
        ▼                           3-factor DuPont, 8 red-flag rules
GOLD    ratios · anomalies
        │  04_register_tools        5 UC SQL functions = the agent's read tools
        ▼
AGENT   fsa/agent_core.py           OpenAI-protocol tool-calling loop (model-
        │                           agnostic FMAPI endpoint), 7 tools total
        ▼
APP     app/                        Streamlit Databricks App: agent chat +
                                    statement visuals + company overview
```

Everything deploys as one asset bundle; the pipeline runs on serverless jobs compute.

## Environments & promotion

Medallion schemas are fixed (`bronze` / `silver` / `gold`); environment isolation is at the **catalog** level:

| Git branch | Bundle target | Catalog | App |
|---|---|---|---|
| `dev` | dev | `fs_analysis_agent_dev` | fsa-agent-dev |
| `test` | test | `fs_analysis_agent_test` | fsa-agent-test |
| `main` | prod | `fs_analysis_agent` | fsa-agent-prod |

Workflow: build on `dev` (CI tests + auto-deploy) → PR `dev → test` for review → PR `test → main` for prod. Only prod runs the monthly refresh schedule. Each environment's app **derives its catalog from its own name** (`fsa-agent-test` → `fs_analysis_agent_test`), so a single `app.yaml` serves all environments.

## The agent

The tool-calling loop (`fsa/agent_core.py`) speaks the OpenAI chat-completions protocol against a Databricks Foundation Model API endpoint. Seven tools:

| Tool | Backing | What it does |
|---|---|---|
| `list_companies` | UC SQL function | What's ingested (availability ground truth) |
| `get_statements` | UC SQL function | Extracted IS/BS/CF line items per period |
| `get_ratios` | UC SQL function | All precomputed ratios incl. DuPont |
| `get_anomalies` | UC SQL function | Red flags — the ONLY anomalies it may report |
| `search_mdna` | UC SQL function | Keyword search over MD&A narrative |
| `ingest_company` | Jobs API | Triggers the pipeline for a new ticker (validated against SEC's registry; guarded against re-ingesting) |
| `get_pipeline_status` | Jobs API | Is new data ready yet? |

Guardrails learned the hard way (all encoded in prompt + loop): numbers only from tool results; anomalies only from `get_anomalies`; availability must be re-verified with fresh tool calls each turn (models happily parrot their own stale claims from chat history); textual pseudo-tool-calls (`[get_ratios(...)]`, leaked `/ipython` template tokens) are detected and the model is pushed back to real tool calls; per-call timeouts so nothing hangs silently.

**Model choice:** Free Edition workspaces sit in a trust tier that rate-limits premium hosted models (Claude, GPT-5) to zero, so the default endpoint is `databricks-gpt-oss-120b` (Llama 4 Maverick was tried first and hallucinated entire tool transcripts as prose). On a paid workspace, set the `llm_endpoint` bundle variable to `databricks-claude-sonnet-5` — same loop, no code change.

**Why no Model Serving endpoint:** the loop runs in-process inside the Databricks App — no always-on resource, Free-Edition-friendly. `src/05_agent.py` runs the same agent in a notebook with MLflow 3 tracing, and documents the production path (`mlflow.pyfunc.log_model` + `agents.deploy`).

## On-demand, incremental ingestion

Ask about a company that isn't ingested and the agent doesn't guess — `ingest_company` validates the ticker, then triggers the pipeline via the Jobs API (the app's service principal holds `CAN_MANAGE_RUN` through a bundle resource binding). Ingestion is **incremental**: only the requested ticker is fetched from EDGAR; bronze writes are per-ticker merges (delete + append), so existing companies are untouched and a new ticker lands in about a minute plus the silver/gold rebuild. The scheduled monthly run passes `include_existing=true` to refresh every company. A run that fails to ingest an explicitly requested ticker **fails loudly**, so the agent reports it instead of silently re-triggering.

## Data extraction highlights

**XBRL tag fallbacks** — companies tag the same concept differently (`Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax` vs banks' `RevenuesNetOfInterestExpense`); every line item carries an ordered fallback list, including financial-sector tags (`NoninterestExpense`, compensation). Periods are trimmed to each company's latest 5 fiscal years / 8 discrete quarters (per-tag selection would otherwise resurrect stale periods).

**Segment revenue from inline XBRL** — Products/Services or business-segment revenue is dimensional XBRL that the JSON APIs don't return; it's parsed best-effort from the 10-K HTML itself (`fsa/segments.py`: contexts + `ix:nonFraction` facts).

**MD&A extraction** — heuristic Item 7 boundary detection on the filing HTML (latest-start-wins beats the table of contents), chunked for keyword search. The agent cross-checks red flags against management's own explanation.

## The app (three tabs)

**🤖 Ask the analyst** — the agent chat, with a live audit trail of every tool call. The company you ask about becomes the default selection on the other tabs.

**📊 Ratios & red flags** — KPI cards with YoY deltas; an income statement **Sankey** (segment revenue → revenue → gross profit/COGS → operating income/opex → R&D/SG&A → pre-tax → net income/tax, labels show $ and % of revenue, explicit column layout, bank-shaped variant for financial-sector filers, waterfall fallback for loss-makers) with a **full-year / latest-quarter toggle**; a common-size analysis table (% of revenue, YoY, QoQ); six 5-year trend charts that adapt to what the company actually reports; and a red-flag heatmap (severity × fiscal year × count).

**🏢 Company overview** — SEC profile (sector, HQ, exchange), Wikipedia introduction, 5-year price chart with market cap (Stooq primary, Yahoo fallback), top-3 news headlines with links (Google News RSS), **market competitors** (Yahoo industry top-5 by size, SEC browse-by-SIC fallback) plus a peer-comparison table from ingested same-SIC companies, and top institutional holders. All keyless sources.

## Red-flag rules (`fsa/anomalies.py`)

Receivables outrunning revenue (aggressive revenue recognition), profit not backed by operating cash flow (accruals), inventory building faster than COGS, margin compression during growth, current ratio < 1, leverage spikes, negative equity, revenue decline. Each flag carries severity, the metric values it fired on, and a plain-English explanation.

## Run it

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run fsa_pipeline -t dev   # ingest → statements → ratios → tools
# then open the app (Compute → Apps), or run src/05_agent.py for MLflow tracing
```

Day-to-day you rarely run these by hand — pushing to `dev`/`test`/`main` deploys the matching target via GitHub Actions. Local unit tests (no Databricks needed): `python tests/test_local.py`.

**Per-environment one-time setup:** run the pipeline once (creates the catalog and grants the owner `ALL PRIVILEGES + MANAGE` — CI-created catalogs are otherwise invisible to humans), then grant the environment's **app service principal**: `USE CATALOG` on the catalog and `USE SCHEMA, SELECT` on bronze/silver (+ `EXECUTE` on gold).

## CI/CD

`ci.yml` runs the offline test suite (synthetic XBRL through extraction, ratios, DuPont, anomaly rules, MD&A and segment parsing, agent plumbing — 22 checks) plus a drift check that `app/agent_core.py` stays in sync with `fsa/agent_core.py`. `databricks-cd.yml` validates bundles on PRs and deploys on push: `dev`→dev, `test`→test, `main`→prod, authenticating as a service principal via `DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET` repo secrets. Deploys never auto-run the pipeline.

## Repo map

| Path | What it is |
|---|---|
| `fsa/edgar.py` | SEC EDGAR client (rate-limited, retries, fair-access headers) |
| `fsa/statements.py` | XBRL tag→line-item mapping with ordered fallbacks, period trimming |
| `fsa/segments.py` | Segment revenue from inline XBRL (contexts + facts, best-effort) |
| `fsa/mdna.py` | MD&A extraction from filing HTML + chunking |
| `fsa/ratios.py` | Ratio + DuPont computation (pure pandas, unit-tested) |
| `fsa/anomalies.py` | Deterministic red-flag rules |
| `fsa/agent_core.py` | The tool-calling loop (copied into `app/`, CI-checked) |
| `src/01–04_*.py` | Pipeline notebooks (bronze → silver → gold → UC tools) |
| `src/05_agent.py` | Agent in a notebook with MLflow 3 tracing + deploy notes |
| `app/` | Streamlit Databricks App (chat + visuals + overview) |
| `tests/test_local.py` | Offline tests on synthetic data |

## Notes & limits

Annual 10-K focus with discrete-quarter 10-Q support (cumulative YTD periods are filtered out). Financial-sector filers get a bank-shaped statement flow; concepts they don't report (inventory, current ratio) are shown as "not reported" rather than blank charts. Segment parsing is best-effort by design. Data is as-reported US-GAAP, USD only. Free Edition specifics that shaped the design: serverless-only compute, no metastore-admin editing (hence pipeline-managed grants), premium LLMs rate-limited to zero (hence open models), and no always-on serving endpoints (hence the in-app agent loop).
