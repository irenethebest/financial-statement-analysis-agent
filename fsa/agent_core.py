"""The agent loop — shared by the dev notebook and the Databricks App.

Design principle: **the LLM never computes a number.** All math lives in
deterministic pipeline code and Unity Catalog functions; Claude decides
which tool to call and narrates the JSON that comes back.

The loop speaks the OpenAI chat-completions protocol, which Databricks
Foundation Model APIs expose natively — so it is model-agnostic: Llama 4
Maverick on Free Edition (Claude/GPT are trust-tier gated there), Claude
on paid workspaces. One config variable, no code change.
"""

from __future__ import annotations

import json
import re
from typing import Callable

# Some open models write tool calls as prose — "[get_ratios(...)]", bare
# "get_ratios(p_ticker=X)", or leaked chat-template tokens like "/ipython" —
# instead of emitting native tool calls. Detect it so the loop can correct
# course rather than return a hallucinated transcript to the user.
_FAKE_TOOL_CALL = re.compile(
    r"(?:get_statements|get_ratios|get_anomalies|list_companies"
    r"|ingest_company|get_pipeline_status)\s*\("
    r"|/ipython|<\|python_tag\|>|<\|eom\|>")


def is_fake_tool_text(text: str) -> bool:
    """True if a message narrates tool calls instead of making them."""
    return bool(_FAKE_TOOL_CALL.search(text or ""))


def content_to_text(content) -> str:
    """Normalize message content to a plain string.

    Reasoning models (e.g. gpt-oss) return content as a list of typed
    parts rather than a string; keep only the text parts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in (None, "text", "output_text"):
                    t = p.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            else:  # SDK objects
                if getattr(p, "type", "text") in ("text", "output_text"):
                    t = getattr(p, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        return "\n".join(x for x in parts if x).strip()
    return str(content)

SYSTEM_PROMPT = """\
You are a financial statement analyst. You explain SEC filings to smart
people who are NOT accountants.

Hard rules:
- Every number you state must come verbatim from a tool result. Never
  estimate, extrapolate, or compute numbers yourself.
- Only report red flags returned by get_anomalies. If it returns an empty
  array, say no rules fired — do not invent concerns.
- If a value is missing/null, say the company didn't report it that way
  rather than guessing.
- Call tools ONLY through the tool-calling mechanism. Never write tool
  names, function-call syntax, or bracketed calls like [get_ratios(...)]
  in your prose.
- Gather ALL the data you need first (statements, ratios, anomalies),
  THEN write the answer. Your final message must be the complete,
  finished analysis — never a promise of further analysis.
- Only companies returned by list_companies are analyzable right now.
  If asked about any other company, call ingest_company ONCE with its
  ticker — this starts the governed ingestion pipeline (takes a few
  minutes). Then tell the user ingestion has started and to ask again
  shortly. Never fabricate analysis while data is being ingested. If the
  user asks whether it's ready, call get_pipeline_status.

When asked to analyze a company, follow this shape:
1. One-paragraph plain-English verdict up front.
2. What the business earned and owns (revenue, profit, cash — trend over
   the available years, in round numbers a reader can hold in their head).
3. Financial health: liquidity and leverage, translated (e.g. "for every
   $1 due in the next year it holds $1.60 in short-term assets").
4. DuPont ROE story: is return on equity driven by margins, by asset
   efficiency, or by leverage? Explain what that mix means.
5. Red flags from get_anomalies, each explained in one plain sentence,
   ordered by severity. Include the 'so what'.
6. Close with 2-3 things a curious reader should watch next year.

Define every technical term in parentheses the first time you use it.
Prefer "$94 billion" over "$94,036,000,000".
"""

# OpenAI-format tool specs. Names/signatures mirror the UC functions
# registered by src/04_register_tools.py — COMMENTs there and descriptions
# here must stay in sync.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "List companies available for analysis (ticker, CIK, entity name). Call first if unsure which tickers exist.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statements",
            "description": "Extracted financial-statement line items for one company across recent fiscal periods (USD, as reported to the SEC).",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                    "p_statement": {"type": "string", "enum": ["IS", "BS", "CF"],
                                    "description": "IS=income statement, BS=balance sheet, CF=cash flow"},
                },
                "required": ["p_ticker", "p_statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ratios",
            "description": "All precomputed ratios for one company: liquidity, leverage, profitability, 3-factor DuPont ROE decomposition, efficiency days, cash quality, YoY growth (fractions). Report as-is; never recalculate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                },
                "required": ["p_ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": "Rule-based earnings-quality red flags (severity info/medium/high) with plain-English explanations. The ONLY source of anomalies you may report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                },
                "required": ["p_ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_company",
            "description": "Start the governed ingestion pipeline (SEC EDGAR -> bronze -> silver -> gold) for a US-listed SEC filer that is NOT yet in list_companies. Takes a few minutes; existing companies are preserved. Call at most once per company, then tell the user to ask again shortly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_ticker": {"type": "string",
                                 "description": "Stock ticker to ingest, e.g. APLD"},
                },
                "required": ["p_ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_status",
            "description": "Check whether the ingestion pipeline is currently running or when it last finished. Use when the user asks if newly requested data is ready.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Tools implemented client-side (Jobs API), not as UC SQL functions.
ACTION_TOOLS = {"ingest_company", "get_pipeline_status"}

# Positional parameter order of each UC function (SQL UDFs are positional).
_PARAM_ORDER = {
    "list_companies": [],
    "get_statements": ["p_ticker", "p_statement"],
    "get_ratios": ["p_ticker"],
    "get_anomalies": ["p_ticker"],
}


def build_tool_sql(fq_schema: str, name: str, args: dict) -> str:
    """SELECT statement invoking the UC function for a tool call.

    `fq_schema` is `catalog.schema`. String args are quote-escaped —
    the LLM's arguments are untrusted input.
    """
    if name not in _PARAM_ORDER:
        raise KeyError(f"Unknown tool: {name}")
    rendered = []
    for p in _PARAM_ORDER[name]:
        v = str(args.get(p, ""))
        rendered.append("'" + v.replace("'", "''") + "'")
    return f"SELECT {fq_schema}.{name}({', '.join(rendered)}) AS result"


def run_agent(
    client,
    model: str,
    user_message: str,
    execute_tool: Callable[[str, dict], str],
    history: list[dict] | None = None,
    max_turns: int = 10,
    on_event: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    """Tool-calling loop.

    client        OpenAI-compatible client (Databricks FMAPI or Anthropic-
                  compatible gateway).
    execute_tool  (tool_name, args) -> JSON string. The caller decides how
                  UC functions are executed (spark.sql in notebooks, SQL
                  warehouse in the app).
    on_event      optional progress callback for UIs.

    Returns (final_text, full_message_list) — the message list is the
    audit trail: every tool call and result the answer is based on.
    """
    notify = on_event or (lambda _msg: None)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": user_message})

    for _turn in range(max_turns):
        notify(f"Thinking (step {_turn + 1}/{max_turns})…")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SPECS,
            max_tokens=4000,
            temperature=0.2,  # analysis, not creativity — stay on rails
            timeout=120,      # never hang silently on a single LLM call
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            content = content_to_text(msg.content)
            if _FAKE_TOOL_CALL.search(content) and _turn < max_turns - 1:
                # The model narrated tool calls instead of making them.
                notify("Model wrote tool syntax as text — nudging it to "
                       "use real tool calls")
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Nothing was executed — you wrote tool-call syntax "
                        "as text. Invoke tools through the tool-calling "
                        "mechanism now, or give your complete final answer "
                        "using only data already retrieved."),
                })
                continue
            return content, messages + [
                {"role": "assistant", "content": content}
            ]

        messages.append({
            "role": "assistant",
            "content": content_to_text(msg.content),
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            notify(f"Calling tool {name}({args})")
            try:
                result = execute_tool(name, args)
                if result is None or result == "":
                    result = "[]"
            except Exception as exc:  # surface errors so the agent can adapt
                result = json.dumps({"error": str(exc)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return ("I hit the maximum number of analysis steps before finishing. "
            "Partial work is in the trace above."), messages
