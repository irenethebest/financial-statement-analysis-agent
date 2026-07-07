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
    wide = query(f"SELECT * FROM {CATALOG}.silver.statements_wide "
                 "ORDER BY ticker, period_end")
    return companies, ratios, anomalies, wide


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
GOOD = "#1a9c6b"
TEAL = "#1aa098"


def _b(v) -> str:
    """$ in human units."""
    if v is None or pd.isna(v):
        return "—"
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"${v / div:,.1f}{suf}"
    return f"${v:,.0f}"


def _fy(period_end: str) -> str:
    return f"FY{str(period_end)[:4]}"


def build_income_flow(row) -> "go.Figure | None":
    """Sankey of the latest income statement. Returns None when flows are
    non-positive/missing (loss-makers) — caller falls back to a waterfall."""
    import plotly.graph_objects as go

    rev, cogs = row.get("revenue"), row.get("cost_of_revenue")
    gp, oi = row.get("gross_profit"), row.get("operating_income")
    ni = row.get("net_income")
    tax = row.get("income_tax_expense")
    pti = row.get("pretax_income")

    core = [rev, cogs, gp, oi, ni]
    if any(v is None or pd.isna(v) or v <= 0 for v in core):
        return None

    labels, links = [], []  # links: (src, dst, value, color)

    def node(name, value=None):
        text = f"{name}<br>{_b(value)}" if value is not None else name
        labels.append(text)
        return len(labels) - 1

    n_rev = node("Revenue", rev)
    n_gp = node("Gross profit", gp)
    n_cogs = node("Cost of revenue", cogs)
    n_oi = node("Operating income", oi)
    n_opex = node("Operating expenses", gp - oi)
    links += [(n_rev, n_gp, gp, "rgba(26,156,107,.45)"),
              (n_rev, n_cogs, cogs, "rgba(210,72,63,.35)"),
              (n_gp, n_oi, oi, "rgba(26,156,107,.45)"),
              (n_gp, n_opex, max(gp - oi, 0), "rgba(224,136,58,.35)")]

    if pd.notna(pti) and pd.notna(tax) and pti > 0 and ni > 0 and tax >= 0:
        n_pti = node("Pre-tax income", pti)
        n_ni = node("Net income", ni)
        n_tax = node("Income tax", tax)
        if pti >= oi:  # non-operating income added
            n_oth = node("Non-operating income", pti - oi)
            links += [(n_oi, n_pti, oi, "rgba(26,156,107,.45)"),
                      (n_oth, n_pti, max(pti - oi, 1e-9),
                       "rgba(47,128,237,.35)")]
        else:  # non-operating costs (interest etc.)
            n_oth = node("Non-operating costs", oi - pti)
            links += [(n_oi, n_pti, pti, "rgba(26,156,107,.45)"),
                      (n_oi, n_oth, oi - pti, "rgba(224,136,58,.35)")]
        links += [(n_pti, n_ni, ni, "rgba(26,156,107,.6)"),
                  (n_pti, n_tax, max(tax, 1e-9), "rgba(91,58,160,.35)")]
    else:
        n_ni = node("Net income", ni)
        links.append((n_oi, n_ni, min(ni, oi), "rgba(26,156,107,.6)"))

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=18, thickness=16,
                  color=NAVY, line=dict(width=0)),
        link=dict(source=[l[0] for l in links],
                  target=[l[1] for l in links],
                  value=[max(l[2], 1e-9) for l in links],
                  color=[l[3] for l in links]),
    ))
    fig.update_layout(height=420, margin=dict(t=20, b=20, l=10, r=10),
                      font_size=13)
    return fig


def build_income_waterfall(row) -> "go.Figure":
    """Waterfall fallback — handles losses gracefully."""
    import plotly.graph_objects as go

    steps = [("Revenue", row.get("revenue"), "absolute"),
             ("Cost of revenue", -(row.get("cost_of_revenue") or 0), "relative"),
             ("Operating expenses",
              -((row.get("gross_profit") or 0) - (row.get("operating_income") or 0)),
              "relative")]
    oi, pti = row.get("operating_income"), row.get("pretax_income")
    if pd.notna(pti) and pd.notna(oi):
        steps.append(("Non-operating, net", pti - oi, "relative"))
    if pd.notna(row.get("income_tax_expense")):
        steps.append(("Income tax", -row["income_tax_expense"], "relative"))
    steps.append(("Net income", row.get("net_income"), "total"))

    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps],
        y=[s[1] for s in steps],
        measure=[s[2] for s in steps],
        text=[_b(s[1]) for s in steps], textposition="outside",
        increasing=dict(marker_color=GOOD),
        decreasing=dict(marker_color=BAD),
        totals=dict(marker_color=NAVY),
        connector=dict(line=dict(color="#999", width=1)),
    ))
    fig.update_layout(height=420, margin=dict(t=30, b=20), showlegend=False)
    return fig


with tab_data:
    try:
        companies, ratios, anomalies, wides = load_overview()
    except Exception as exc:
        st.error(f"Could not load tables from {FQ}: {exc}")
        st.stop()

    tickers = sorted(companies["ticker"].tolist())
    sel = st.selectbox("Company", tickers)
    r = ratios[ratios["ticker"] == sel].sort_values("period_end").copy()
    a = anomalies[anomalies["ticker"] == sel].copy()
    w = wides[wides["ticker"] == sel].sort_values("period_end").copy()
    r["fy"] = r["period_end"].map(_fy)
    w["fy"] = w["period_end"].map(_fy)

    # ---- KPI cards with YoY deltas ----
    last = r.iloc[-1] if not r.empty else None
    prev = r.iloc[-2] if len(r) > 1 else None
    wlast = w.iloc[-1] if not w.empty else None

    def _delta_pp(cur, pre):
        return (f"{(cur - pre) * 100:+.1f} pp"
                if pre is not None and pd.notna(cur) and pd.notna(pre) else None)

    if last is not None and wlast is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Revenue", _b(wlast["revenue"]),
                  f"{last['revenue_growth']:+.1%} YoY"
                  if pd.notna(last.get("revenue_growth")) else None)
        c2.metric("Net income", _b(wlast["net_income"]),
                  f"{last['net_income_growth']:+.1%} YoY"
                  if pd.notna(last.get("net_income_growth")) else None)
        c3.metric("ROE", f"{last['roe']:.1%}" if pd.notna(last["roe"]) else "—",
                  _delta_pp(last["roe"], prev["roe"] if prev is not None else None))
        c4.metric("Net margin",
                  f"{last['net_margin']:.1%}" if pd.notna(last["net_margin"]) else "—",
                  _delta_pp(last["net_margin"],
                            prev["net_margin"] if prev is not None else None))
        n_last_fy = len(a[a["period_end"] == last["period_end"]])
        c5.metric("Red flags (latest FY)", n_last_fy,
                  delta=None, help="Rule-based flags fired for the most "
                                   "recent fiscal year")

    sub_flow, sub_trend, sub_flags, sub_raw = st.tabs(
        ["🌊 Income statement flow", "📈 5-year trends", "🚩 Red flags",
         "🗂 Raw data"])

    # ---- Sankey / waterfall ----
    with sub_flow:
        if wlast is None:
            st.info("No statement data.")
        else:
            st.caption(f"{sel} — {_fy(wlast['period_end'])} "
                       f"(fiscal year ended {wlast['period_end']})")
            fig = build_income_flow(wlast)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sankey needs positive flows — this company had "
                        "losses or missing items, showing a waterfall "
                        "instead (negatives welcome).")
                st.plotly_chart(build_income_waterfall(wlast),
                                use_container_width=True)

    # ---- Trends ----
    with sub_trend:
        import plotly.graph_objects as go

        t1, t2 = st.columns(2)
        with t1:
            st.subheader("Earnings")
            fig = go.Figure()
            for col, name, color in [("revenue", "Revenue", NAVY),
                                     ("operating_income", "Operating income", ACCENT),
                                     ("net_income", "Net income", GOOD)]:
                fig.add_bar(x=w["fy"], y=w[col], name=name, marker_color=color)
            fig.update_layout(barmode="group", height=320,
                              margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Margins")
            fig = go.Figure()
            for col, name, color in [("gross_margin", "Gross", NAVY),
                                     ("operating_margin", "Operating", ACCENT),
                                     ("net_margin", "Net", GOOD)]:
                fig.add_scatter(x=r["fy"], y=r[col], name=name, mode="lines+markers",
                                line=dict(color=color))
            fig.update_layout(height=300, yaxis_tickformat=".0%",
                              margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cash conversion cycle (days)")
            fig = go.Figure()
            fig.add_bar(x=r["fy"], y=r["dso_days"], name="DSO (collect)",
                        marker_color=ACCENT)
            fig.add_bar(x=r["fy"], y=r["dio_days"], name="DIO (hold inventory)",
                        marker_color=WARN)
            fig.add_bar(x=r["fy"], y=-r["dpo_days"], name="DPO (pay suppliers)",
                        marker_color=TEAL)
            fig.add_scatter(x=r["fy"], y=r["cash_conversion_cycle_days"],
                            name="CCC", mode="lines+markers",
                            line=dict(color=NAVY, width=3))
            fig.update_layout(barmode="relative", height=300,
                              margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.subheader("Returns & efficiency")
            fig = go.Figure()
            fig.add_scatter(x=r["fy"], y=r["roe"], name="ROE",
                            mode="lines+markers", line=dict(color=NAVY, width=3))
            fig.add_scatter(x=r["fy"], y=r["roa"], name="ROA",
                            mode="lines+markers", line=dict(color=GOOD))
            fig.add_scatter(x=r["fy"], y=r["dupont_asset_turnover"],
                            name="Asset turnover (x)", mode="lines+markers",
                            line=dict(color=WARN, dash="dot"))
            fig.update_layout(height=320, margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Receivables vs revenue growth")
            st.caption("When the orange bar tops the blue one, the "
                       "ar_outpaces_revenue red flag territory begins.")
            fig = go.Figure()
            fig.add_bar(x=r["fy"], y=r["revenue_growth"], name="Revenue growth",
                        marker_color=ACCENT)
            fig.add_bar(x=r["fy"], y=r["accounts_receivable_growth"],
                        name="AR growth", marker_color=WARN)
            fig.update_layout(barmode="group", height=300,
                              yaxis_tickformat=".0%", margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Profit vs cash: NI and OCF")
            fig = go.Figure()
            fig.add_bar(x=w["fy"], y=w["net_income"], name="Net income",
                        marker_color=NAVY)
            fig.add_bar(x=w["fy"], y=w["operating_cash_flow"],
                        name="Operating cash flow", marker_color=GOOD)
            fig.update_layout(barmode="group", height=300,
                              margin=dict(t=10, b=10),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

    # ---- Red flags ----
    with sub_flags:
        if a.empty:
            st.success("No anomaly rules fired for this company.")
        else:
            sev_score = {"info": 1, "medium": 2, "high": 3}
            hm = a.copy()
            hm["score"] = hm["severity"].map(sev_score)
            hm["fy"] = hm["period_end"].map(_fy)
            grid = hm.pivot_table(index="rule_id", columns="fy",
                                  values="score", aggfunc="max")
            all_fy = sorted(r["fy"].unique())
            grid = grid.reindex(columns=all_fy)
            fig = px.imshow(
                grid,
                color_continuous_scale=[[0, "#e8edf3"], [0.33, "#5b9bd5"],
                                        [0.66, WARN], [1.0, BAD]],
                zmin=0, zmax=3, aspect="auto",
                labels=dict(color="severity"))
            fig.update_layout(height=90 + 42 * len(grid),
                              coloraxis_showscale=False,
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔵 info · 🟠 medium · 🔴 high — hover a cell for "
                       "the rule and year")

            badge = {"high": "🔴", "medium": "🟠", "info": "🔵"}
            for _, row in a.sort_values(
                    ["period_end", "severity"], ascending=[False, True]).iterrows():
                st.markdown(
                    f"{badge.get(row['severity'], '•')} **{row['rule_id']}** "
                    f"({_fy(row['period_end'])}) — {row['explanation']}")

    # ---- Raw ----
    with sub_raw:
        st.subheader("All ratios")
        st.dataframe(r.drop(columns=["fy"]), use_container_width=True)
        st.subheader("Statement line items (wide)")
        st.dataframe(w.drop(columns=["fy"]), use_container_width=True)
