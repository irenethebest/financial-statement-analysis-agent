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
  LLM_ENDPOINT              FMAPI endpoint (default databricks-claude-sonnet-5)
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
LLM = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-5")

st.set_page_config(page_title="Financial Statement Analysis Agent",
                   page_icon="📄", layout="wide")

NAVY, ACCENT, BAD, WARN = "#0f2742", "#2f80ed", "#d2483f", "#e0883a"


# ---------------------------------------------------------------------------
# Databricks connections (cached per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _sql_conn():
    from databricks import sql
    from databricks.sdk.core import Config

    cfg = Config()  # app's OAuth service principal
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: cfg.authenticate,
    )


@st.cache_resource(show_spinner=False)
def _llm_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient().serving_endpoints.get_open_ai_client()


def query(sql_text: str) -> pd.DataFrame:
    with _sql_conn().cursor() as cur:
        cur.execute(sql_text)
        return cur.fetchall_arrow().to_pandas()


def execute_tool(name: str, args: dict) -> str:
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

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input(
        "e.g. Analyze AAPL — is its profit backed by real cash?")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            status = st.status("Analyzing…", expanded=True)
            try:
                answer, trail = agent_core.run_agent(
                    client=_llm_client(),
                    model=LLM,
                    user_message=prompt,
                    execute_tool=execute_tool,
                    on_event=lambda msg: status.write(msg),
                )
                status.update(label="Done — audit trail above", state="complete",
                              expanded=False)
                st.markdown(answer)
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
