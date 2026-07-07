"""FSA Agent — Streamlit Databricks App.

The agent loop runs *inside this app* (Free-Edition-friendly: no dedicated
Model Serving endpoint). The app's service principal:
  * queries gold/silver tables and executes UC tool functions via the
    bound SQL warehouse,
  * calls the pay-per-token Foundation Model API (Claude) for the loop.

Env (set by databricks.yml):
  DATABRICKS_WAREHOUSE_ID   bound SQL warehouse
  APP_CATALOG               environment catalog (fs_analysis_agent[_dev|_test]);
                            schemas are fixed: bronze / silver / gold
  LLM_ENDPOINT              FMAPI endpoint (default databricks-llama-4-maverick;
                            Free Edition gates Claude/GPT endpoints to rate
                            limit 0 — use open models there)
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# In the repo, use the shared fsa package; when deployed as a Databricks
# App only this folder is shipped, so fall back to the local copy
# (app/agent_core.py — keep in sync with fsa/agent_core.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from fsa import agent_core  # noqa: E402
except ImportError:
    import agent_core  # noqa: E402

CATALOG = os.getenv("APP_CATALOG", "fs_analysis_agent_dev")
FQ = f"{CATALOG}.gold"  # tool functions live in the gold (serving) schema
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
LLM = os.getenv("LLM_ENDPOINT", "databricks-llama-4-maverick")
PIPELINE_JOB_ID = os.getenv("PIPELINE_JOB_ID")  # for the ingest_company tool
UA_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "irenejinheechoi@gmail.com")

st.set_page_config(page_title="Financial Statement Analysis Agent",
                   page_icon="📄", layout="wide")

NAVY, ACCENT, BAD, WARN = "#0f2742", "#2f80ed", "#d2483f", "#e0883a"


# ---------------------------------------------------------------------------
# Databricks connections (cached per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _ws():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()  # auto-auths as the app's service principal


@st.cache_resource(show_spinner=False)
def _llm_client():
    return _ws().serving_endpoints.get_open_ai_client()


def query(sql_text: str, timeout_s: int = 150) -> pd.DataFrame:
    """Run SQL on the bound warehouse via the Statement Execution API.

    REST-based (no connector handshake), explicit timeout, and surfaces
    server-side errors (e.g. PERMISSION_DENIED) as readable exceptions.
    """
    import time

    from databricks.sdk.service.sql import StatementState

    w = _ws()
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql_text,
        wait_timeout="30s",
    )
    deadline = time.time() + timeout_s
    while (resp.status.state in (StatementState.PENDING, StatementState.RUNNING)
           and time.time() < deadline):
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)

    if resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status.error else resp.status.state
        raise RuntimeError(f"SQL failed ({resp.status.state}): {err}")

    cols = [c.name for c in resp.manifest.schema.columns]
    df = pd.DataFrame(resp.result.data_array or [], columns=cols)
    # The API returns strings; restore numerics where the whole column casts.
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any() and (conv.isna() == df[c].isna()).all():
            df[c] = conv
    return df


def _ingest_company(ticker: str) -> str:
    """Agent action tool: validate the ticker against SEC's company list,
    then trigger the governed pipeline job with it (merge semantics)."""
    import json

    import requests

    if not PIPELINE_JOB_ID:
        return json.dumps({"error": "Ingestion is not configured "
                                    "(PIPELINE_JOB_ID missing)."})
    if not ticker or not ticker.isalnum():
        return json.dumps({"error": f"Invalid ticker: {ticker!r}"})

    # Hard guard: if the company is already ingested, never re-trigger —
    # tell the agent to analyze instead (models don't always check first).
    try:
        n = query(f"SELECT count(*) AS n FROM {CATALOG}.bronze.companies "
                  f"WHERE ticker = '{ticker}'").iloc[0, 0]
        if int(n) > 0:
            return json.dumps({
                "status": "already_available",
                "note": f"{ticker} is already ingested. Do NOT ingest again "
                        "— call get_ratios / get_anomalies / get_statements "
                        "now and answer the user's question.",
            })
    except Exception:
        pass  # if the check fails, fall through to normal flow

    # Validate against SEC's official ticker list before burning a job run.
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": f"FSA agent {UA_EMAIL}"}, timeout=20)
        known = {e["ticker"].upper(): e["title"] for e in resp.json().values()}
        if ticker not in known:
            return json.dumps({"error": f"{ticker} is not an SEC-registered "
                                        "ticker. Ask the user to verify it."})
        title = known[ticker]
    except Exception:
        title = None  # validation is best-effort; proceed

    w = _ws()
    # Don't stack runs — if the pipeline is already running, report that.
    active = list(w.jobs.list_runs(job_id=int(PIPELINE_JOB_ID),
                                   active_only=True, limit=1))
    if active:
        return json.dumps({"status": "already_running",
                           "note": "The pipeline is already running; new "
                                   "tickers can be requested once it finishes."})

    run = w.jobs.run_now(job_id=int(PIPELINE_JOB_ID),
                         notebook_params={"tickers": ticker})
    return json.dumps({
        "status": "started",
        "ticker": ticker,
        "company": title,
        "run_id": run.run_id,
        "note": "Governed ingestion started (EDGAR -> bronze -> silver -> "
                "gold). Typically ready in a few minutes; the user should "
                "ask about this company again then.",
    })


def _pipeline_status() -> str:
    import json

    if not PIPELINE_JOB_ID:
        return json.dumps({"error": "PIPELINE_JOB_ID missing."})
    w = _ws()
    runs = list(w.jobs.list_runs(job_id=int(PIPELINE_JOB_ID), limit=1))
    if not runs:
        return json.dumps({"status": "never_run"})
    r = runs[0]
    state = r.state.life_cycle_state.value if r.state else "UNKNOWN"
    result = (r.state.result_state.value
              if r.state and r.state.result_state else None)
    done = state in ("TERMINATED", "INTERNAL_ERROR")
    return json.dumps({
        "running": not done,
        "life_cycle_state": state,
        "result_state": result,
        "note": ("Finished — newly ingested companies are queryable now."
                 if done and result == "SUCCESS" else
                 "Still running — data not ready yet." if not done else
                 "Last run did not succeed."),
    })


def execute_tool(name: str, args: dict) -> str:
    if name == "ingest_company":
        # Refresh the company list cache afterwards so the UI catches up.
        load_overview.clear()
        return _ingest_company(str(args.get("p_ticker", "")).upper().strip())
    if name == "get_pipeline_status":
        load_overview.clear()
        return _pipeline_status()
    df = query(agent_core.build_tool_sql(FQ, name, args))
    val = df.iloc[0, 0]
    return val if val is not None else "[]"


@st.cache_data(ttl=600, show_spinner=False)
def load_overview():
    companies = query(f"SELECT * FROM {CATALOG}.bronze.companies")
    ratios = query(
        f"SELECT * FROM {CATALOG}.gold.ratios ORDER BY ticker, period_end")
    anomalies = query(
        f"SELECT * FROM {CATALOG}.gold.anomalies ORDER BY ticker, period_end")
    return companies, ratios, anomalies


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("📄 Financial Statement Analysis Agent")
st.caption(
    f"SEC EDGAR → Delta medallion pipeline → UC tool functions → "
    f"Claude agent (`{LLM}`). All numbers are computed deterministically; "
    "the LLM only narrates."
)

tab_agent, tab_data = st.tabs(["🤖 Ask the analyst", "📊 Ratios & red flags"])

# ---- Agent chat ----
with tab_agent:
    if "chat" not in st.session_state:
        st.session_state.chat = []  # [(role, text)]

    def md(text: str) -> str:
        # Streamlit markdown treats $...$ as LaTeX; escape dollar amounts.
        return text.replace("$", r"\$")

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(md(text))

    prompt = st.chat_input(
        "e.g. Analyze AAPL — is its profit backed by real cash?")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(md(prompt))
        with st.chat_message("assistant"):
            status = st.status("Analyzing…", expanded=True)
            try:
                # Pass recent chat history so the agent remembers e.g. that
                # it already started an ingestion earlier in the session —
                # but drop any earlier hallucinated tool transcripts so the
                # model doesn't learn the bad pattern from its own output.
                hist = [{"role": r, "content": t}
                        for r, t in st.session_state.chat[:-1][-6:]
                        if not (r == "assistant"
                                and agent_core.is_fake_tool_text(t))]
                answer, trail = agent_core.run_agent(
                    client=_llm_client(),
                    model=LLM,
                    user_message=prompt,
                    execute_tool=execute_tool,
                    history=hist,
                    on_event=lambda msg: status.write(msg),
                )
                status.update(label="Done — audit trail above", state="complete",
                              expanded=False)
                st.markdown(md(answer))
                st.session_state.chat.append(("assistant", answer))
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(f"Agent error: {exc}")

# ---- Data explorer ----
with tab_data:
    try:
        companies, ratios, anomalies = load_overview()
    except Exception as exc:
        st.error(f"Could not load tables from {FQ}: {exc}")
        st.stop()

    tickers = sorted(companies["ticker"].tolist())
    sel = st.selectbox("Company", tickers)
    r = ratios[ratios["ticker"] == sel]
    a = anomalies[anomalies["ticker"] == sel]

    c1, c2, c3, c4 = st.columns(4)
    last = r.iloc[-1] if not r.empty else None
    if last is not None:
        c1.metric("ROE", f"{last['roe']:.1%}" if pd.notna(last["roe"]) else "—")
        c2.metric("Net margin", f"{last['net_margin']:.1%}"
                  if pd.notna(last["net_margin"]) else "—")
        c3.metric("Current ratio", f"{last['current_ratio']:.2f}"
                  if pd.notna(last["current_ratio"]) else "—")
        c4.metric("Red flags (all yrs)", len(a))

    left, right = st.columns(2)
    with left:
        st.subheader("DuPont ROE decomposition")
        dp = r.melt(
            id_vars="period_end",
            value_vars=["dupont_net_margin", "dupont_asset_turnover",
                        "dupont_equity_multiplier"],
            var_name="component", value_name="value")
        fig = px.bar(dp, x="period_end", y="value", color="component",
                     barmode="group",
                     color_discrete_sequence=[NAVY, ACCENT, WARN])
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Cash quality: OCF vs net income")
        fig2 = px.line(r, x="period_end", y="ocf_to_net_income", markers=True,
                       color_discrete_sequence=[ACCENT])
        fig2.add_hline(y=1.0, line_dash="dot", line_color=BAD,
                       annotation_text="cash = reported profit")
        fig2.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Red flags")
    if a.empty:
        st.success("No anomaly rules fired for this company.")
    else:
        badge = {"high": "🔴", "medium": "🟠", "info": "🔵"}
        for _, row in a.iterrows():
            st.markdown(
                f"{badge.get(row['severity'], '•')} **{row['rule_id']}** "
                f"({row['period_end']}) — {row['explanation']}")

    with st.expander("All ratios (raw)"):
        st.dataframe(r, use_container_width=True)
