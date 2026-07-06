# Financial Statement Analysis Agent

An AI agent that reads SEC filings and explains them to non-accountants — built end-to-end on Databricks (Free Edition compatible).

Feed it a ticker; it pulls the company's 10-K data from SEC EDGAR, extracts the three financial statements, computes ~30 ratios including a DuPont ROE decomposition, screens for earnings-quality red flags, and writes a plain-English analysis. The core design rule: **the LLM never computes a number** — all math is deterministic pipeline code and Unity-Catalog-governed SQL functions; Claude only decides which tools to call and narrates the results.

## Architecture

```
SEC EDGAR JSON APIs (no scraping, no key)
        │  01_ingest_edgar          rate-limited client, User-Agent per SEC rules
        ▼
BRONZE  bronze.companies · bronze.xbrl_facts · bronze.filings
        │  02_build_statements      US-GAAP tag fallbacks, period selection,
        ▼                           restatement dedupe (latest filing wins)
SILVER  silver.statements · silver.statements_wide
        │  03_ratios_anomalies      liquidity/leverage/profitability/efficiency,
        ▼                           3-factor DuPont, 8 red-flag rules
GOLD    gold.ratios · gold.anomalies
        │  04_register_tools        4 UC SQL functions in gold = the agent's tools
        ▼
AGENT   fsa/agent_core.py           OpenAI-protocol tool-calling loop against
        │                           a pay-per-token Foundation Model API
        ▼                           endpoint (model-agnostic), MLflow-traced
APP     app/                        Streamlit Databricks App: chat + dashboard
```

Everything deploys as one asset bundle; the pipeline runs on serverless jobs compute.

## Environments & promotion

Medallion schemas are fixed (`bronze` / `silver` / `gold`); environment isolation is at the **catalog** level, one catalog per target:

| Git branch | Bundle target | Catalog |
|---|---|---|
| `dev` | dev | `fs_analysis_agent_dev` |
| `test` | test | `fs_analysis_agent_test` |
| `main` | prod | `fs_analysis_agent` |

Workflow: build on a feature branch (or locally) → push to `dev` (CI tests + auto-deploy to the dev catalog) → validate, then promote to `test` → final check, then merge to `main`, which deploys prod. Only prod runs the monthly refresh schedule.

## Why the agent runs inside the app

On Free Edition (and as cost discipline anywhere), an always-on Model Serving endpoint is the resource to avoid. The tool-calling loop is plain Python speaking the OpenAI protocol, so the Streamlit app executes it in-process: UC tool functions run on the bound SQL warehouse, the LLM is the shared pay-per-token endpoint. `src/05_agent.py` documents the production path (`mlflow.pyfunc.log_model` + `agents.deploy`) for a paid workspace — same loop, different serving skin.

The agent is also **model-agnostic by design**: Free Edition workspaces sit in a trust tier that rate-limits premium hosted models (Claude, GPT) to zero, so the default endpoint is `databricks-llama-4-maverick`; on a paid workspace, setting the `llm_endpoint` bundle variable to `databricks-claude-sonnet-5` swaps the model with no code change.

## Red-flag rules (`fsa/anomalies.py`)

Receivables outrunning revenue (aggressive revenue recognition), profit not backed by operating cash flow (accruals), inventory building faster than COGS, margin compression during growth, current ratio < 1, leverage spikes, negative equity, revenue decline. Each flag carries severity, the metric values it fired on, and a plain-English explanation — the agent may only report flags from this list.

## Run it

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run fsa_pipeline -t dev   # ingest → statements → ratios → tools
# then open the app: fsa-agent-dev (Compute → Apps), or run src/05_agent.py
# in a notebook for the MLflow-traced version
```

In day-to-day work you rarely run these by hand — pushing to `dev`/`test`/`main` deploys the matching target via GitHub Actions. Change companies via the `tickers` bundle variable or the job widget. Local unit tests (no Databricks needed): `python tests/test_local.py`.

## CI/CD

Two GitHub Actions workflows (same pattern as my [accounting_analytics](../accounting_analytics) project):

`ci.yml` runs on every push and PR — no credentials needed. It executes the offline test suite and verifies `app/agent_core.py` hasn't drifted from `fsa/agent_core.py`.

`databricks-cd.yml` validates the asset bundle on PRs and deploys on pushes: `dev` → dev target, `test` → test target, `main` → prod. It authenticates as a workspace service principal via two repo secrets, `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` (OAuth M2M). Deploys never auto-run the pipeline — EDGAR calls and serverless compute only happen on the prod monthly schedule or a manual `bundle run`.

## Repo map

| Path | What it is |
|---|---|
| `fsa/edgar.py` | SEC EDGAR client (rate-limited, retry, fair-access headers) |
| `fsa/statements.py` | XBRL tag→line-item mapping with ordered fallbacks |
| `fsa/ratios.py` | Ratio + DuPont computation (pure pandas, unit-tested) |
| `fsa/anomalies.py` | Deterministic red-flag rules |
| `fsa/agent_core.py` | The tool-calling loop (shared; copied into `app/`) |
| `src/01–04_*.py` | Pipeline notebooks (bronze → silver → gold → UC tools) |
| `src/05_agent.py` | Agent in a notebook with MLflow 3 tracing + deploy notes |
| `app/` | Streamlit Databricks App (chat + ratio dashboard) |
| `tests/test_local.py` | Offline tests on synthetic XBRL data |

## Notes & limits

10-K (annual) focus by default; the extractor supports 10-Q via the `form` parameter (discrete quarters only, cumulative YTD periods are filtered out). Financial-sector companies (banks, insurers) use different statement structures — several ratios will be blank for them by design. Data is as-reported US-GAAP, USD only.
