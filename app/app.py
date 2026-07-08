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

import json
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

    # include_existing=false -> fetch ONLY this ticker; bronze writes are
    # per-ticker merges, so existing companies are untouched (fast).
    run = w.jobs.run_now(job_id=int(PIPELINE_JOB_ID),
                         notebook_params={"tickers": ticker,
                                          "include_existing": "false"})
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
    try:
        wide_q = query(f"SELECT * FROM {CATALOG}.silver.statements_wide_q "
                       "ORDER BY ticker, period_end")
    except Exception:
        wide_q = pd.DataFrame()
    try:
        segs = query(f"SELECT * FROM {CATALOG}.bronze.segment_revenue")
    except Exception:
        segs = pd.DataFrame()
    return companies, ratios, anomalies, wide, wide_q, segs


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("📄 Financial Statement Analysis Agent")
st.caption(
    f"SEC EDGAR → Delta medallion pipeline → UC tool functions → "
    f"Claude agent (`{LLM}`). All numbers are computed deterministically; "
    "the LLM only narrates."
)

tab_agent, tab_data, tab_overview = st.tabs(
    ["🤖 Ask the analyst", "📊 Ratios & red flags", "🏢 Company overview"])

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
                # Remember which company the agent worked on so the
                # Ratios & red flags tab opens on it.
                called = []
                for m in trail:
                    if not isinstance(m, dict):
                        continue
                    for tc in m.get("tool_calls") or []:
                        try:
                            t = json.loads(
                                tc["function"]["arguments"]).get("p_ticker")
                            if t:
                                called.append(str(t).upper().strip())
                        except Exception:
                            pass
                if called:
                    st.session_state.focus_ticker = called[-1]
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


_GREEN = "rgba(26,156,107,.45)"
_GREEN2 = "rgba(26,156,107,.6)"
_RED = "rgba(210,72,63,.35)"
_ORANGE = "rgba(224,136,58,.35)"
_BLUE = "rgba(47,128,237,.35)"
_PURPLE = "rgba(91,58,160,.35)"


def build_income_flow(row, segments=None) -> "go.Figure | None":
    """Sankey of the latest income statement, laid out in explicit columns
    so sibling nodes stack to their parent (Revenue = GP + COGS visually).
    Labels carry $ and % of revenue. Optional left column: revenue by
    segment (from inline-XBRL parsing). Returns None for non-positive
    flows (loss-makers) — caller falls back to a waterfall."""
    import plotly.graph_objects as go

    rev, cogs = row.get("revenue"), row.get("cost_of_revenue")
    gp, oi = row.get("gross_profit"), row.get("operating_income")
    ni = row.get("net_income")
    tax = row.get("income_tax_expense")
    pti = row.get("pretax_income")
    rd, sga = row.get("rd_expense"), row.get("sga_expense")

    comp = row.get("compensation_expense")
    opex_line = row.get("operating_expenses")

    if any(v is None or pd.isna(v) or v <= 0 for v in [rev, ni]):
        return None
    # Standard shape needs COGS/gross/operating income; financial-sector
    # filers (banks) have none of those — use revenue -> expenses -> pretax.
    standard = not any(v is None or pd.isna(v) or v <= 0
                       for v in [cogs, gp, oi])
    fin_ok = (pd.notna(pti) and 0 < pti <= rev and pd.notna(tax)
              and tax >= 0 and ni > 0)
    if not standard and not fin_ok:
        return None

    # nodes: (label, $value, column). links: (src, dst, value, color)
    nodes: list[tuple[str, float, int]] = []
    links: list[tuple[int, int, float, str]] = []
    has_segments = bool(segments)
    col_offset = 1 if has_segments else 0

    def node(name, value, col, rank=9):
        # rank = top-down order within a column; the profit path is rank 0
        # so it hugs the top of every column and ribbons run straight.
        nodes.append([f"{name}<br>{_b(value)} · {value / rev:.0%}",
                      value, col + col_offset, rank])
        return len(nodes) - 1

    n_rev = node("Revenue", rev, 0, 0)

    # Segment column (feeds Revenue). Residual keeps the sum honest.
    if has_segments:
        seg_sum = sum(s["value"] for s in segments)
        shown = [(s["label"], s["value"]) for s in segments]
        if seg_sum < rev * 0.98:
            shown.append(("Other", rev - seg_sum))
        elif seg_sum > rev * 1.05:  # parsed something inconsistent — drop
            shown = []
        for rank_i, (name, v) in enumerate(shown):
            idx = len(nodes)
            nodes.append([f"{name}<br>{_b(v)} · {v / rev:.0%}", v, 0,
                          rank_i])
            links.append((idx, n_rev, max(v, 1e-9), _BLUE))

    if standard:
        n_gp = node("Gross profit", gp, 1, 0)
        n_cogs = node("Cost of revenue", cogs, 1, 5)
        n_oi = node("Operating income", oi, 2, 0)
        opex = gp - oi
        n_opex = node("Operating expenses", opex, 2, 3)
        links += [(n_rev, n_gp, gp, _GREEN),
                  (n_rev, n_cogs, cogs, _RED),
                  (n_gp, n_oi, oi, _GREEN),
                  (n_gp, n_opex, max(opex, 1e-9), _ORANGE)]

        # Opex breakdown lives in the SAME column as the pre-tax node —
        # ribbons never span across a column (that caused the overlaps).
        known = 0.0
        rank_c = 3
        for name, v in (("R&D", rd), ("SG&A", sga)):
            if pd.notna(v) and v > 0 and known + v <= opex * 1.02:
                idx = node(name, v, 3, rank_c)
                links.append((n_opex, idx, v, _ORANGE))
                known += v
                rank_c += 1
        if known > 0 and opex - known > opex * 0.02:
            idx = node("Other opex", opex - known, 3, rank_c)
            links.append((n_opex, idx, opex - known, _ORANGE))

        if pd.notna(pti) and pd.notna(tax) and pti > 0 and ni > 0 and tax >= 0:
            n_pti = node("Pre-tax income", pti, 3, 0)
            if pti >= oi:
                # injection sits directly under Operating income
                n_oth = node("Non-operating income", pti - oi, 2, 1)
                links += [(n_oi, n_pti, oi, _GREEN),
                          (n_oth, n_pti, max(pti - oi, 1e-9), _BLUE)]
            else:
                n_oth = node("Non-operating costs", oi - pti, 3, 1)
                links += [(n_oi, n_pti, pti, _GREEN),
                          (n_oi, n_oth, oi - pti, _ORANGE)]
            n_ni = node("Net income", ni, 4, 0)
            n_tax = node("Income tax", tax, 4, 1)
            links += [(n_pti, n_ni, ni, _GREEN2),
                      (n_pti, n_tax, max(tax, 1e-9), _PURPLE)]
        else:
            n_ni = node("Net income", ni, 3, 0)
            links.append((n_oi, n_ni, min(ni, oi), _GREEN2))
    else:
        # ---- Financial-sector shape:
        # Revenue -> Pre-tax income + Total operating expenses,
        # expenses -> Compensation / Other, pre-tax -> Net income + Tax.
        opex_t = (opex_line if pd.notna(opex_line)
                  and abs(opex_line - (rev - pti)) < rev * 0.15
                  else rev - pti)
        n_pti = node("Pre-tax income", pti, 1, 0)
        n_opex = node("Operating expenses", opex_t, 1, 2)
        links += [(n_rev, n_pti, pti, _GREEN),
                  (n_rev, n_opex, max(opex_t, 1e-9), _ORANGE)]
        if pd.notna(comp) and 0 < comp <= opex_t:
            idx = node("Compensation & benefits", comp, 2, 2)
            links.append((n_opex, idx, comp, _ORANGE))
            if opex_t - comp > opex_t * 0.02:
                idx = node("Other expenses", opex_t - comp, 2, 3)
                links.append((n_opex, idx, opex_t - comp, _ORANGE))
        n_ni = node("Net income", ni, 2, 0)
        n_tax = node("Income tax", max(tax, 1e-9), 2, 1)
        links += [(n_pti, n_ni, ni, _GREEN2),
                  (n_pti, n_tax, max(tax, 1e-9), _PURPLE)]

    # ---- Explicit layout: within each column, nodes sort by rank (profit
    # path on top), thickness is normalized by revenue and top-aligned
    # across columns so ribbons run straight instead of weaving.
    order = sorted(range(len(nodes)),
                   key=lambda i: (nodes[i][2], nodes[i][3]))
    n_cols = max(n[2] for n in nodes) + 1
    xs = [0.0] * len(nodes)
    ys = [0.0] * len(nodes)
    col_cum: dict[int, float] = {}
    for i in order:
        _, v, c, _ = nodes[i]
        cum = col_cum.get(c, 0.0)
        ys[i] = min(0.03 + 0.9 * (cum + v / 2) / rev, 0.97)
        col_cum[c] = cum + v + rev * 0.03  # small gap between nodes
        xs[i] = 0.02 + 0.96 * c / max(n_cols - 1, 1)

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=[n[0] for n in nodes], x=xs, y=ys,
                  pad=14, thickness=16, color=NAVY, line=dict(width=0)),
        link=dict(source=[l[0] for l in links],
                  target=[l[1] for l in links],
                  value=[max(l[2], 1e-9) for l in links],
                  color=[l[3] for l in links]),
    ))
    fig.update_layout(height=460, margin=dict(t=20, b=20, l=10, r=10),
                      font_size=12)
    return fig


def build_income_waterfall(row) -> "go.Figure | None":
    """Waterfall fallback — handles losses and bank-style statements.
    Returns None when there isn't enough data for a meaningful chart."""
    import plotly.graph_objects as go

    rev, ni = row.get("revenue"), row.get("net_income")
    if pd.isna(rev) or pd.isna(ni):
        return None
    cogs, gp = row.get("cost_of_revenue"), row.get("gross_profit")
    oi, pti = row.get("operating_income"), row.get("pretax_income")
    tax = row.get("income_tax_expense")
    opex_line = row.get("operating_expenses")

    steps = [("Revenue", rev, "absolute")]
    if pd.notna(cogs) and pd.notna(gp) and pd.notna(oi):
        steps += [("Cost of revenue", -cogs, "relative"),
                  ("Operating expenses", -(gp - oi), "relative")]
        if pd.notna(pti):
            steps.append(("Non-operating, net", pti - oi, "relative"))
    elif pd.notna(pti):  # bank-style: one total-expenses step
        opex_t = (opex_line if pd.notna(opex_line)
                  and abs(opex_line - (rev - pti)) < abs(rev) * 0.15
                  else rev - pti)
        steps.append(("Operating expenses", -opex_t, "relative"))
    elif pd.notna(opex_line):
        steps.append(("Operating expenses", -opex_line, "relative"))
    else:
        return None
    if pd.notna(tax):
        steps.append(("Income tax", -tax, "relative"))
    steps.append(("Net income", ni, "total"))

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
        companies, ratios, anomalies, wides, wides_q, seg_df = load_overview()
    except Exception as exc:
        st.error(f"Could not load tables from {FQ}: {exc}")
        st.stop()

    tickers = sorted(companies["ticker"].tolist())
    # Streamlit ignores `index` once a keyed widget has state, so when the
    # agent's focus changes we set the widget state directly (both tabs).
    focus = st.session_state.get("focus_ticker")
    if focus in tickers and st.session_state.get("_focus_applied") != focus:
        st.session_state["data_ticker"] = focus
        st.session_state["overview_ticker"] = focus
        st.session_state["_focus_applied"] = focus
    if "data_ticker" not in st.session_state:
        st.session_state["data_ticker"] = (focus if focus in tickers
                                           else tickers[0])
    if st.session_state["data_ticker"] not in tickers:
        st.session_state["data_ticker"] = tickers[0]
    sel = st.selectbox("Company", tickers, key="data_ticker",
                       help="Follows the company you last asked the "
                            "analyst about")
    def _recent(df, years=5.2):
        """Defense-in-depth: keep only periods within `years` of the
        company's latest period (stale tag periods pollute the x-axis)."""
        if df.empty or "period_end" not in df.columns:
            return df
        ends = pd.to_datetime(df["period_end"])
        return df[ends >= ends.max() - pd.Timedelta(days=int(years * 365))]

    r = _recent(ratios[ratios["ticker"] == sel]) \
        .sort_values("period_end").copy()
    a = anomalies[anomalies["ticker"] == sel]
    a = a[a["period_end"].isin(set(r["period_end"]))].copy()
    w = _recent(wides[wides["ticker"] == sel]) \
        .sort_values("period_end").copy()
    wq = (_recent(wides_q[wides_q["ticker"] == sel], years=2.3)
          .sort_values("period_end").copy()
          if not wides_q.empty else pd.DataFrame())
    sel_segments = (
        [{"label": row["label"], "value": row["value"]}
         for _, row in seg_df[seg_df["ticker"] == sel].iterrows()]
        if not seg_df.empty else [])
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

    sub_stmt, sub_flags, sub_raw = st.tabs(
        ["📊 Income statement & trends", "🚩 Red flags", "🗂 Raw data"])
    sub_flow = sub_trend = sub_stmt  # merged page, two sections

    # ---- Sankey / waterfall ----
    with sub_flow:
        if wlast is None:
            st.info("No statement data.")
        else:
            st.subheader("Income statement flow")

            # Offer the latest quarter too when a 10-Q is NEWER than the
            # last fiscal year end (we already ingest quarterly data).
            flow_row, flow_segments = wlast, sel_segments
            flow_is_q = False
            flow_label = (f"{_fy(wlast['period_end'])} (fiscal year ended "
                          f"{wlast['period_end']}, 10-K)")
            qrow = wq.iloc[-1] if len(wq) > 0 else None
            if (qrow is not None
                    and str(qrow["period_end"]) > str(wlast["period_end"])):
                choice = st.radio(
                    "Period",
                    [f"Full year — {_fy(wlast['period_end'])}",
                     f"Latest quarter — ended {qrow['period_end']}"],
                    horizontal=True, key="flow_period")
                if choice.startswith("Latest quarter"):
                    flow_row = qrow
                    flow_is_q = True
                    flow_segments = []  # segments are parsed from the 10-K
                    flow_label = (f"quarter ended {qrow['period_end']} "
                                  "(10-Q, more recent than the last "
                                  "fiscal year)")

            st.caption(f"{sel} — {flow_label}; labels show $ and % of "
                       "revenue"
                       + (" · left column = revenue by source (parsed "
                          "from the 10-K's inline XBRL)"
                          if flow_segments else ""))
            fig = build_income_flow(flow_row, flow_segments)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                wf = build_income_waterfall(flow_row)
                if wf is not None:
                    st.info("Sankey needs positive flows — this period had "
                            "losses or missing items, showing a waterfall "
                            "instead (negatives welcome).")
                    st.plotly_chart(wf, use_container_width=True)
                else:
                    st.info("Not enough extracted line items to draw the "
                            "income statement flow for this period.")

            # ---- Common-size / growth analysis ----
            st.subheader("Income statement analysis")
            _IS_ROWS = [
                ("revenue", "Revenue"),
                ("cost_of_revenue", "Cost of revenue"),
                ("gross_profit", "Gross profit"),
                ("rd_expense", "R&D expense"),
                ("sga_expense", "SG&A expense"),
                ("operating_income", "Operating income"),
                ("pretax_income", "Pre-tax income"),
                ("income_tax_expense", "Income tax"),
                ("net_income", "Net income"),
            ]
            wprev = w.iloc[-2] if len(w) > 1 else None
            qlast = wq.iloc[-1] if len(wq) > 0 else None
            qprev = wq.iloc[-2] if len(wq) > 1 else None

            def _pct(cur, pre):
                if (cur is None or pre is None or pd.isna(cur)
                        or pd.isna(pre) or pre == 0):
                    return "—"
                return f"{(cur - pre) / abs(pre):+.1%}"

            # Follow the FY / latest-quarter toggle above.
            if flow_is_q:
                base = qrow
                # same quarter last year = 4 discrete quarters back
                prev_yoy = wq.iloc[-5] if len(wq) >= 5 else None
                prev_qoq = qprev
                col_hdr = f"Q ended {qrow['period_end']} ($)"
            else:
                base = wlast
                prev_yoy = wprev
                prev_qoq = None  # FY view: QoQ uses the two latest quarters
                col_hdr = f"{_fy(wlast['period_end'])} ($)"

            table = []
            for col, name in _IS_ROWS:
                v = base.get(col)
                if v is None or pd.isna(v):
                    continue
                if flow_is_q:
                    qoq = _pct(v, prev_qoq.get(col)
                               if prev_qoq is not None else None)
                else:
                    qoq = _pct(qlast.get(col)
                               if qlast is not None else None,
                               qprev.get(col)
                               if qprev is not None else None)
                table.append({
                    "Line item": name,
                    col_hdr: _b(v),
                    "% of revenue": (f"{v / base['revenue']:.1%}"
                                     if pd.notna(base.get("revenue"))
                                     and base["revenue"] else "—"),
                    "YoY %": _pct(v, prev_yoy.get(col)
                                  if prev_yoy is not None else None),
                    "QoQ %": qoq,
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True,
                         hide_index=True)
            if flow_is_q:
                st.caption("Latest-quarter view — YoY compares to the same "
                           "quarter last year"
                           + (f" ({prev_yoy['period_end']})"
                              if prev_yoy is not None else
                              " (not available: needs 5+ ingested quarters)")
                           + "; QoQ to the prior quarter"
                           + (f" ({qprev['period_end']})."
                              if qprev is not None else "."))
            elif qlast is not None:
                st.caption(f"QoQ compares the two most recent discrete "
                           f"quarters ({qprev['period_end'] if qprev is not None else '—'} "
                           f"→ {qlast['period_end']}) from 10-Q filings.")
            st.divider()
            st.subheader("5-year trends")

    # ---- Trends ----
    with sub_trend:
        import plotly.graph_objects as go

        _SKIP_NOTE = ("Not reported by this company — financial-sector "
                      "filers use a different statement structure.")

        def _cols_with_data(df_, specs):
            return [(c, n, col) for c, n, col in specs
                    if c in df_.columns and df_[c].notna().any()]

        def _bar_chart(df_, specs, title, tickformat=None, caption=None):
            st.subheader(title)
            specs = _cols_with_data(df_, specs)
            if not specs:
                st.caption(_SKIP_NOTE)
                return
            if caption:
                st.caption(caption)
            fig = go.Figure()
            for c, n, color in specs:
                fig.add_bar(x=df_["fy"], y=df_[c], name=n, marker_color=color)
            fig.update_layout(barmode="group", height=300,
                              margin=dict(t=10, b=10),
                              yaxis_tickformat=tickformat,
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

        def _line_chart(df_, specs, title, tickformat=None):
            st.subheader(title)
            specs = _cols_with_data(df_, specs)
            if not specs:
                st.caption(_SKIP_NOTE)
                return
            fig = go.Figure()
            for c, n, color in specs:
                fig.add_scatter(x=df_["fy"], y=df_[c], name=n,
                                mode="lines+markers", line=dict(color=color))
            fig.update_layout(height=300, margin=dict(t=10, b=10),
                              yaxis_tickformat=tickformat,
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

        t1, t2 = st.columns(2)
        with t1:
            _bar_chart(w, [("revenue", "Revenue", NAVY),
                           ("operating_income", "Operating income", ACCENT),
                           ("net_income", "Net income", GOOD)],
                       "Earnings")
            _line_chart(r, [("gross_margin", "Gross", NAVY),
                            ("operating_margin", "Operating", ACCENT),
                            ("net_margin", "Net", GOOD)],
                        "Margins", tickformat=".0%")

            st.subheader("Cash conversion cycle (days)")
            if r["dso_days"].notna().any() or r["dio_days"].notna().any():
                fig = go.Figure()
                fig.add_bar(x=r["fy"], y=r["dso_days"], name="DSO (collect)",
                            marker_color=ACCENT)
                fig.add_bar(x=r["fy"], y=r["dio_days"],
                            name="DIO (hold inventory)", marker_color=WARN)
                fig.add_bar(x=r["fy"], y=-r["dpo_days"],
                            name="DPO (pay suppliers)", marker_color=TEAL)
                fig.add_scatter(x=r["fy"], y=r["cash_conversion_cycle_days"],
                                name="CCC", mode="lines+markers",
                                line=dict(color=NAVY, width=3))
                fig.update_layout(barmode="relative", height=300,
                                  margin=dict(t=10, b=10),
                                  legend=dict(orientation="h", y=1.15))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(_SKIP_NOTE)

        with t2:
            _line_chart(r, [("roe", "ROE", NAVY),
                            ("roa", "ROA", GOOD),
                            ("dupont_asset_turnover", "Asset turnover (x)",
                             WARN)],
                        "Returns & efficiency")
            _bar_chart(r, [("revenue_growth", "Revenue growth", ACCENT),
                           ("accounts_receivable_growth", "AR growth", WARN)],
                       "Receivables vs revenue growth", tickformat=".0%",
                       caption="When the orange bar tops the blue one, "
                               "ar_outpaces_revenue red-flag territory "
                               "begins.")
            _bar_chart(w, [("net_income", "Net income", NAVY),
                           ("operating_cash_flow", "Operating cash flow",
                            GOOD)],
                       "Profit vs cash: NI and OCF")

    # ---- Red flags ----
    with sub_flags:
        if a.empty:
            st.success("No anomaly rules fired for this company "
                       "(last 5 fiscal years).")
        else:
            hm = a.copy()
            hm["fy"] = hm["period_end"].map(_fy)
            grid = hm.pivot_table(index="severity", columns="fy",
                                  values="rule_id", aggfunc="count")
            all_fy = sorted(r["fy"].unique())
            sev_order = [s for s in ("high", "medium", "info")
                         if s in grid.index]
            grid = grid.reindex(index=sev_order, columns=all_fy)
            fig = px.imshow(
                grid, text_auto=True, aspect="auto",
                color_continuous_scale=[[0, "#e8edf3"], [0.5, WARN],
                                        [1.0, BAD]],
                labels=dict(x="fiscal year", y="severity",
                            color="flags"))
            fig.update_layout(height=110 + 60 * len(grid),
                              coloraxis_showscale=False,
                              margin=dict(t=10, b=10))
            fig.update_traces(
                hovertemplate="FY %{x} · %{y}: %{z} flag(s)<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Count of red flags by severity and fiscal year — "
                       "details below")

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


# ---------------------------------------------------------------------------
# Company overview — profile (SEC), price (Stooq), intro (Wikipedia),
# news (Google News RSS), holders (yfinance best-effort), peers (own catalog)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_profiles() -> pd.DataFrame:
    try:
        df = query(f"SELECT * FROM {CATALOG}.bronze.company_profile")
    except Exception:
        return pd.DataFrame()
    # query() auto-numerifies columns; codes must stay strings (SIC "6199",
    # zip, fiscal year end "0928" with its leading zero).
    for c in df.columns:
        if c != "cik":
            df[c] = (df[c].astype("string")
                     .str.replace(r"\.0$", "", regex=True))
    if "fiscal_year_end" in df.columns:
        df["fiscal_year_end"] = df["fiscal_year_end"].str.zfill(4)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def price_history(ticker: str) -> tuple[pd.DataFrame, str]:
    """Daily closes, ~5y. Stooq first (keyless), yfinance fallback.
    Returns (df[Date, Close], source_name)."""
    import io

    import requests
    try:
        resp = requests.get(
            f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d",
            headers={"User-Agent": "Mozilla/5.0 (FSA portfolio app)"},
            timeout=20)
        df = pd.read_csv(io.StringIO(resp.text))
        if "Close" in df.columns and len(df) > 10:
            df["Date"] = pd.to_datetime(df["Date"])
            return df[["Date", "Close"]].tail(5 * 252), "Stooq"
    except Exception:
        pass
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        if h is not None and not h.empty:
            df = h.reset_index()[["Date", "Close"]]
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df, "Yahoo Finance"
    except Exception:
        pass
    return pd.DataFrame(), ""


@st.cache_data(ttl=86400, show_spinner=False)
def wiki_summary(name: str) -> dict:
    """SEC entity names are ALL-CAPS ('APPLE INC', '.../DE'), and the
    Wikipedia summary endpoint is title-case-sensitive — so resolve the
    real page title via the search API first."""
    import re as _re
    import urllib.parse

    import requests
    headers = {"User-Agent": f"FSA portfolio app {UA_EMAIL}"}
    # Drop SEC state-of-incorporation suffixes: '/DE', '/OH/', ...
    clean = _re.sub(r"/[A-Z]{2}/?$", "", str(name)).strip()
    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "format": "json",
                    "srlimit": 1, "srsearch": f"{clean} company"},
            headers=headers, timeout=15).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return {}
        title = hits[0]["title"]
        resp = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title),
            headers=headers, timeout=15)
        if resp.status_code == 200:
            j = resp.json()
            if j.get("extract") and j.get("type") == "standard":
                return {"text": j["extract"],
                        "url": j.get("content_urls", {})
                                .get("desktop", {}).get("page")}
    except Exception:
        pass
    return {}


@st.cache_data(ttl=1800, show_spinner=False)
def news_headlines(name: str, n: int = 3) -> list[dict]:
    import urllib.parse
    import xml.etree.ElementTree as ET

    import requests
    try:
        q = urllib.parse.quote(f'"{name}" stock')
        resp = requests.get(
            f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
            timeout=15)
        root = ET.fromstring(resp.content)
        out = []
        for item in root.iter("item"):
            out.append({
                "title": item.findtext("title") or "",
                "link": item.findtext("link") or "",
                "date": (item.findtext("pubDate") or "")[:16],
            })
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def sec_industry_companies(sic: str) -> tuple[pd.DataFrame, str]:
    """All SEC filers under an exact SIC code (official browse-edgar
    endpoint, Atom output — keyless). Returns (df, error_message)."""
    import html as _html
    import re as _re

    import requests
    try:
        resp = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={"action": "getcompany", "SIC": sic, "type": "10-K",
                    "count": "100", "output": "atom"},
            headers={"User-Agent": f"FSA portfolio app {UA_EMAIL}"},
            timeout=20)
        if resp.status_code in (429, 503):  # SEC rate/robot gate — retry
            import time as _time
            for backoff in (1, 3):
                _time.sleep(backoff)
                resp = requests.get(resp.url, headers={
                    "User-Agent": f"FSA portfolio app {UA_EMAIL}"},
                    timeout=20)
                if resp.status_code == 200:
                    break
        if resp.status_code != 200:
            return pd.DataFrame(), f"SEC returned HTTP {resp.status_code}"
        rows = []
        for e in _re.findall(r"<entry>(.*?)</entry>", resp.text, _re.S):
            t = _re.search(r"<title>(.*?)</title>", e, _re.S)
            c = _re.search(r"CIK=(\d{4,10})", e)
            if t:
                name = _html.unescape(t.group(1)).strip()
                name = _re.sub(r"\s*\(\d{7,10}\)\s*$", "", name)
                rows.append({"company": name,
                             "cik": int(c.group(1)) if c else None})
        if not rows:
            return pd.DataFrame(), "no entries parsed from SEC response"
        return (pd.DataFrame(rows).drop_duplicates(subset=["company"]), "")
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=86400, show_spinner=False)
def cik_ticker_map() -> dict[int, str]:
    import requests
    try:
        data = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": f"FSA portfolio app {UA_EMAIL}"},
            timeout=20).json()
        return {int(e["cik_str"]): e["ticker"] for e in data.values()}
    except Exception:
        return {}


def _yf_cap(t):
    try:
        import yfinance as yf
        return yf.Ticker(t).fast_info["marketCap"]
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def industry_top5(sic: str, ticker: str) -> tuple[pd.DataFrame, str, str]:
    """Top 5 companies in the selected company's industry, ranked by size.

    Route 1 (primary): Yahoo's industry classification — one call returns
    the ranked top-companies list directly.
    Route 2 (fallback): SEC browse-by-SIC roster (cloud IPs often get 503
    there) -> CIK->ticker -> parallel Yahoo market caps.
    Returns (df[company,ticker,market_cap], source_note, error)."""
    from concurrent.futures import ThreadPoolExecutor

    errs = []
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        key, ind_name = info.get("industryKey"), info.get("industry")
        if key:
            tc = yf.Industry(key).top_companies
            if tc is not None and not tc.empty:
                tc = tc.reset_index().head(5)
                sym_col = ("symbol" if "symbol" in tc.columns
                           else tc.columns[0])
                name_col = "name" if "name" in tc.columns else tc.columns[1]
                with ThreadPoolExecutor(max_workers=5) as ex:
                    caps = list(ex.map(_yf_cap, tc[sym_col]))
                df = pd.DataFrame({"company": tc[name_col],
                                   "ticker": tc[sym_col],
                                   "market_cap": caps})
                return df, f"Yahoo industry: {ind_name}", ""
        errs.append("no Yahoo industry mapping")
    except Exception as exc:
        errs.append(f"Yahoo route: {type(exc).__name__}")

    ind, err = sec_industry_companies(sic)
    if ind.empty:
        errs.append(f"SEC route: {err}")
        return pd.DataFrame(), "", "; ".join(errs)
    m = cik_ticker_map()
    ind = ind.copy()
    ind["ticker"] = ind["cik"].map(m)
    listed = ind.dropna(subset=["ticker"]).head(120)
    if listed.empty:
        return pd.DataFrame(), "", "no listed tickers in SEC roster"
    with ThreadPoolExecutor(max_workers=8) as ex:
        listed["market_cap"] = list(ex.map(_yf_cap, listed["ticker"]))
    top = (listed.dropna(subset=["market_cap"])
           .sort_values("market_cap", ascending=False).head(5))
    if top.empty:
        return pd.DataFrame(), "", "market caps unavailable (Yahoo)"
    return (top[["company", "ticker", "market_cap"]],
            f"SIC {sic} — {len(ind)} SEC filers", "")


@st.cache_data(ttl=86400, show_spinner=False)
def yf_holders(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).institutional_holders
        if df is not None and not df.empty:
            return df.head(8)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def shares_outstanding(ticker: str) -> float | None:
    try:
        df = query(
            f"SELECT val FROM {CATALOG}.bronze.xbrl_facts "
            f"WHERE ticker = '{ticker}' AND taxonomy = 'dei' "
            "AND tag = 'EntityCommonStockSharesOutstanding' "
            "ORDER BY `end` DESC LIMIT 1")
        return float(df.iloc[0, 0]) if not df.empty else None
    except Exception:
        return None


with tab_overview:
    profiles = load_profiles()
    try:
        companies_o, ratios_o, _, wides_o, _, _ = load_overview()
    except Exception as exc:
        st.error(f"Could not load tables: {exc}")
        st.stop()

    tickers_o = sorted(companies_o["ticker"].tolist())
    focus_o = st.session_state.get("focus_ticker")
    if "overview_ticker" not in st.session_state:
        st.session_state["overview_ticker"] = (
            focus_o if focus_o in tickers_o else tickers_o[0])
    if st.session_state["overview_ticker"] not in tickers_o:
        st.session_state["overview_ticker"] = tickers_o[0]
    sel_o = st.selectbox("Company", tickers_o, key="overview_ticker",
                         help="Follows the company you last asked the "
                              "analyst about")

    prof = (profiles[profiles["ticker"] == sel_o].iloc[0]
            if not profiles.empty
            and (profiles["ticker"] == sel_o).any() else None)
    name = (prof["entity_name"] if prof is not None
            else companies_o.set_index("ticker")
            .loc[sel_o, "entity_name"])

    st.title(name)
    if prof is not None:
        def _p(key):
            v = prof.get(key)
            return None if v is None or pd.isna(v) or v == "" else v

        hq = ", ".join(x for x in [_p("hq_city"), _p("hq_state")] if x)
        bits = [f"**Sector:** {_p('sic_description') or '—'} "
                f"(SIC {_p('sic') or '—'})",
                f"**HQ:** {hq or '—'}",
                f"**Exchange:** {_p('exchange') or '—'}",
                f"**Fiscal year end:** {_p('fiscal_year_end') or '—'}"]
        if _p("website"):
            bits.append(f"[{_p('website')}]({_p('website')})")
        st.markdown(" · ".join(bits))
    else:
        st.caption("Profile not ingested yet — re-run the pipeline to "
                   "populate bronze.company_profile.")

    colL, colR = st.columns([2, 3])

    with colL:
        st.subheader("About")
        wiki = wiki_summary(name)
        if wiki:
            st.write(wiki["text"])
            if wiki.get("url"):
                st.caption(f"[Wikipedia]({wiki['url']})")
        else:
            st.caption("No introduction found.")

        st.subheader("Recent headlines")
        for h in news_headlines(name):
            st.markdown(f"- [{h['title']}]({h['link']})  \n"
                        f"  <sub>{h['date']}</sub>",
                        unsafe_allow_html=True)
        if not news_headlines(name):
            st.caption("No headlines found.")

    with colR:
        px_df, px_src = price_history(sel_o)
        if not px_df.empty:
            last_close = px_df["Close"].iloc[-1]
            yr = px_df[px_df["Date"] >= px_df["Date"].max()
                       - pd.Timedelta(days=365)]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last close", f"${last_close:,.2f}",
                      f"{last_close / px_df['Close'].iloc[-2] - 1:+.2%}"
                      if len(px_df) > 1 else None)
            m2.metric("52w high", f"${yr['Close'].max():,.2f}")
            m3.metric("52w low", f"${yr['Close'].min():,.2f}")
            sh = shares_outstanding(sel_o)
            m4.metric("Market cap", _b(last_close * sh) if sh else "—",
                      help="Last close × latest reported shares "
                           "outstanding (SEC dei)")

            figp = px.area(px_df, x="Date", y="Close",
                           color_discrete_sequence=[ACCENT])
            figp.update_layout(
                height=330, margin=dict(t=10, b=10),
                xaxis=dict(rangeselector=dict(buttons=[
                    dict(count=3, label="3m", step="month",
                         stepmode="backward"),
                    dict(count=1, label="1y", step="year",
                         stepmode="backward"),
                    dict(label="5y", step="all")])))
            st.plotly_chart(figp, use_container_width=True)
            st.caption(f"Daily close (EOD), source: {px_src}.")
        else:
            st.caption("Price history unavailable (Stooq and Yahoo "
                       "Finance both failed — possibly network egress "
                       "or rate limits).")

    st.divider()
    st.subheader("Market competitors")
    if prof is not None and not profiles.empty:
        sic = str(prof.get("sic") or "")
        # Ingested peers: exact industry only (full 4-digit SIC match).
        peer_prof = (profiles[(profiles["sic"] == sic)
                              & (profiles["ticker"] != sel_o)]
                     if sic else pd.DataFrame())
        # Industry leaders, ranked by size (Yahoo primary, SEC fallback).
        with st.spinner("Finding the industry's largest companies "
                        "(first load per industry is slow, then cached)…"):
            top5, note, err = industry_top5(sic, sel_o)
        if not top5.empty:
            disp = pd.DataFrame({
                "Company": top5["company"].values,
                "Ticker": top5["ticker"].values,
                "Market cap": top5["market_cap"].map(_b).values,
                "In catalog": [("✅" if t in tickers_o else "")
                               for t in top5["ticker"]],
            })
            st.markdown(f"**Top 5 in this industry by size** — {note}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption("Sizes from Yahoo Finance. Ask the analyst to "
                       "ingest any of them for the comparison below.")
        else:
            st.caption(f"Industry list unavailable — {err}")

        if peer_prof.empty:
            st.caption(f"No ingested company shares this exact industry "
                       f"(SIC {sic}) yet — ask the analyst to ingest a "
                       "competitor from the list above.")
        else:
            st.markdown(f"**Peer comparison from your data** — "
                        f"same industry (SIC {sic}).")
            latest_r = (ratios_o.sort_values("period_end")
                        .groupby("ticker").tail(1)
                        .set_index("ticker"))
            latest_w = (wides_o.sort_values("period_end")
                        .groupby("ticker").tail(1)
                        .set_index("ticker"))
            rows = []
            for _, p in pd.concat(
                    [profiles[profiles["ticker"] == sel_o],
                     peer_prof]).iterrows():
                t = p["ticker"]
                rr = latest_r.loc[t] if t in latest_r.index else {}
                ww = latest_w.loc[t] if t in latest_w.index else {}
                rows.append({
                    "Ticker": t + (" ⭐" if t == sel_o else ""),
                    "Company": p["entity_name"],
                    "Revenue": _b(ww.get("revenue")),
                    "Net margin": (f"{rr.get('net_margin'):.1%}"
                                   if pd.notna(rr.get("net_margin"))
                                   else "—"),
                    "ROE": (f"{rr.get('roe'):.1%}"
                            if pd.notna(rr.get("roe")) else "—"),
                    "D/E": (f"{rr.get('debt_to_equity'):.2f}"
                            if pd.notna(rr.get("debt_to_equity")) else "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    st.divider()
    st.subheader("Ownership")
    st.caption("Largest institutional shareholders of "
               f"{sel_o} — unrelated to the peer comparison above. "
               "Source: Yahoo Finance (best-effort).")
    hold = yf_holders(sel_o)
    if hold.empty:
        st.caption("Holder data unavailable right now.")
    else:
        st.dataframe(hold, use_container_width=True, hide_index=True)
