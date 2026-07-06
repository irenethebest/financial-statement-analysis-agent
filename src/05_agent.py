# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · The Agent — Claude + UC Tools, Traced with MLflow
# MAGIC
# MAGIC Runs the tool-calling agent (`fsa/agent_core.py`) against Databricks'
# MAGIC pay-per-token **Foundation Model API** (`databricks-claude-sonnet-5`),
# MAGIC executing the Unity Catalog tool functions from notebook 04.
# MAGIC
# MAGIC **MLflow 3 tracing** captures every LLM call and tool execution, so a
# MAGIC reviewer can audit exactly which numbers the summary is built on.
# MAGIC
# MAGIC *Free Edition note:* the agent is served **inside the Databricks App**
# MAGIC (`app/`) rather than a dedicated Model Serving endpoint — no always-on
# MAGIC resource, no quota pressure. The production path (`mlflow.log_model` +
# MAGIC `agents.deploy`) is sketched at the bottom for reference.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow-skinny[databricks] openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "fs_analysis_agent_dev", "Unity Catalog")
dbutils.widgets.text("llm_endpoint", "databricks-claude-sonnet-5",
                     "Foundation Model endpoint")
dbutils.widgets.text("question",
                     "Analyze AAPL's most recent fiscal years.",
                     "Question for the agent")

CATALOG = dbutils.widgets.get("catalog")
FQ = f"{CATALOG}.gold"  # tool functions live in the gold (serving) schema
LLM = dbutils.widgets.get("llm_endpoint")
QUESTION = dbutils.widgets.get("question")

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fsa import agent_core  # noqa: E402

# COMMAND ----------

# MAGIC %md ### Wire the pieces: FMAPI client + UC tool executor + tracing

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

mlflow.openai.autolog()  # traces every LLM call (inputs, outputs, latency)
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}"
                      "/fsa_agent")

client = WorkspaceClient().serving_endpoints.get_open_ai_client()


def execute_tool(name: str, args: dict) -> str:
    """Run a UC function via Spark SQL and hand the JSON back to the LLM."""
    sql = agent_core.build_tool_sql(FQ, name, args)
    return spark.sql(sql).first()["result"] or "[]"

# COMMAND ----------

# MAGIC %md ### Run

# COMMAND ----------

with mlflow.start_run(run_name="fsa_agent_notebook"):
    with mlflow.start_span(name="fsa_agent") as span:
        span.set_inputs({"question": QUESTION, "model": LLM})
        answer, trail = agent_core.run_agent(
            client=client,
            model=LLM,
            user_message=QUESTION,
            execute_tool=execute_tool,
            on_event=print,
        )
        span.set_outputs({"answer": answer,
                          "n_messages": len(trail)})
    mlflow.log_param("llm_endpoint", LLM)
    mlflow.log_metric("n_tool_calls",
                      sum(1 for m in trail if m.get("role") == "tool"))
    mlflow.log_text(answer, "report.md")

print("=" * 78)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production path (reference — not run on Free Edition)
# MAGIC
# MAGIC On a paid workspace you'd package this agent as an MLflow `ChatAgent`,
# MAGIC register it to Unity Catalog, and deploy with Mosaic AI:
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from databricks import agents
# MAGIC
# MAGIC logged = mlflow.pyfunc.log_model(
# MAGIC     name="fsa_agent",
# MAGIC     python_model="agent_as_code.py",      # ChatAgent wrapping run_agent()
# MAGIC     resources=[                            # auto-auth for the endpoint+tools
# MAGIC         DatabricksServingEndpoint(endpoint_name="databricks-claude-sonnet-5"),
# MAGIC         DatabricksFunction(function_name=f"{FQ}.get_ratios"),
# MAGIC         DatabricksFunction(function_name=f"{FQ}.get_anomalies"),
# MAGIC         DatabricksFunction(function_name=f"{FQ}.get_statements"),
# MAGIC         DatabricksFunction(function_name=f"{FQ}.list_companies"),
# MAGIC     ],
# MAGIC     registered_model_name=f"{FQ}.fsa_agent",
# MAGIC )
# MAGIC agents.deploy(f"{FQ}.fsa_agent", logged.registered_model_version)
# MAGIC ```
# MAGIC
# MAGIC That gives you a REST endpoint with AI Gateway governance, the Review
# MAGIC App for feedback, and production tracing — the same loop, different
# MAGIC serving skin.
