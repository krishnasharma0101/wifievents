import io
import json
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st


# ------------------------
# UI CONFIG
# ------------------------
st.set_page_config(page_title="CSV Chat with Local LLM", layout="wide")
st.title("CSV Chat with Local LLM (LM Studio)")


# ------------------------
# LLM Defaults (avoid re-specifying each call)
# ------------------------
LM_ENDPOINT_DEFAULT = "http://localhost:1234"
LM_MODEL_DEFAULT = "openai/gpt-oss-20b"
LM_TEMPERATURE_DEFAULT = 0.7
LM_MAX_TOKENS_DEFAULT = -1
LM_STREAM_DEFAULT = False


# ------------------------
# Session State Init
# ------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "sample_rows" not in st.session_state:
    st.session_state.sample_rows = None
if "column_info" not in st.session_state:
    st.session_state.column_info = None
if "last_code" not in st.session_state:
    st.session_state.last_code = ""
if "http_session" not in st.session_state:
    st.session_state.http_session = requests.Session()


# ------------------------
# Helpers: Schema and Samples
# ------------------------
def extract_schema_and_samples(df: pd.DataFrame, sample_rows: int = 5) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    # Use head(sample_rows) for lightweight context
    sample_df = df.head(sample_rows)
    # Convert to list of dicts with safe JSON-serializable values
    sample_records: List[Dict[str, Any]] = json.loads(sample_df.to_json(orient="records", date_format="iso"))
    return dtypes, sample_records


def build_prompt(user_question: str, column_info: Dict[str, str], sample_rows: List[Dict[str, Any]]) -> str:
    # Keep the prompt stable and structured
    schema_lines = [f"- {col}: {dtype}" for col, dtype in column_info.items()]
    schema_text = "\n".join(schema_lines)
    sample_text = json.dumps(sample_rows, ensure_ascii=False, indent=2)[:5000]

    system_instructions = (
        "You are a senior data analyst. You write concise, correct, and executable Pandas code to answer questions about a DataFrame named df. "
        "You will only return Python code inside a single fenced code block. Do not include any explanations or text outside the code block. "
        "Use Streamlit 'st' to display results (e.g., st.dataframe, st.bar_chart, st.write). "
        "Never read files, never use network access, and do not import new libraries beyond pandas/numpy/streamlit. "
        "Assume df is already loaded. Prefer vectorized Pandas operations."
    )

    task = (
        "Given the following schema and sample rows of df, answer the user's question by returning only executable Python code in a single fenced code block. "
        "Ensure the code runs as-is in a Streamlit app environment with variables df, pd, np, and st already available. "
        "If plotting, prefer simple bar charts with st.bar_chart when appropriate."
    )

    example = (
        "Example format you must follow (no surrounding prose):\n\n"
        "```python\n"
        "# Compute some result using df\n"
        "result = df.head(10)\n"
        "st.dataframe(result)\n"
        "```"
    )

    prompt = (
        f"{system_instructions}\n\n"
        f"{task}\n\n"
        f"Schema (column: dtype):\n{schema_text}\n\n"
        f"Sample rows (truncated JSON):\n{sample_text}\n\n"
        f"User question: {user_question}\n\n"
        f"{example}\n"
    )
    return prompt


# ------------------------
# LLM Client
# ------------------------
def query_llm(
    prompt: str,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    # Use stable defaults so we don't "reload" configs repeatedly
    endpoint = endpoint or LM_ENDPOINT_DEFAULT
    model = model or LM_MODEL_DEFAULT
    temperature = LM_TEMPERATURE_DEFAULT if temperature is None else temperature
    max_tokens = LM_MAX_TOKENS_DEFAULT if max_tokens is None else max_tokens
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You only return Python code inside one code block. No explanations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": LM_STREAM_DEFAULT,
    }
    try:
        resp = st.session_state.http_session.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or ""
    except Exception as e:
        return f"ERROR_CALLING_LLM: {e}"


def query_llm_text(
    prompt: str,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    endpoint = endpoint or LM_ENDPOINT_DEFAULT
    model = model or LM_MODEL_DEFAULT
    temperature = LM_TEMPERATURE_DEFAULT if temperature is None else temperature
    max_tokens = LM_MAX_TOKENS_DEFAULT if max_tokens is None else max_tokens
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful data analyst. Return concise markdown insights. Do not return code blocks."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": LM_STREAM_DEFAULT,
    }
    try:
        resp = st.session_state.http_session.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or ""
    except Exception as e:
        return f"ERROR_CALLING_LLM: {e}"


# ------------------------
# Code Block Extraction
# ------------------------
CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_block(text: str) -> Optional[str]:
    if not text:
        return None
    matches = CODE_BLOCK_REGEX.findall(text)
    if not matches:
        return None
    # Use the last block under the assumption corrections might append
    code = matches[-1].strip()
    return code if code else None


# ------------------------
# Sandboxed Execution with Retries
# ------------------------
def execute_user_code(
    code: str,
    df: pd.DataFrame,
    extra_globals: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    # Restricted globals; provide common analysis libs
    sandbox_globals: Dict[str, Any] = {
        "__builtins__": {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
        },
        "pd": pd,
        "np": np,
        "st": st,
        "df": df,
    }
    if extra_globals:
        sandbox_globals.update(extra_globals)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, sandbox_globals, None)
        std_out = stdout_buffer.getvalue()
        std_err = stderr_buffer.getvalue()
        combined = (std_out + ("\n" + std_err if std_err else "")).strip()
        return True, combined, sandbox_globals
    except Exception:
        tb = traceback.format_exc()
        return False, tb, sandbox_globals
    finally:
        stdout_buffer.close()
        stderr_buffer.close()


# ------------------------
# Sidebar: Settings
# ------------------------
with st.sidebar:
    st.header("LLM Settings")
    endpoint = st.text_input("LM Studio Endpoint", value=LM_ENDPOINT_DEFAULT)
    model = st.text_input("Model", value=LM_MODEL_DEFAULT)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(LM_TEMPERATURE_DEFAULT), step=0.05)
    st.markdown("Enter the local LM Studio endpoint and model name.")


# ------------------------
# File Upload and Preview
# ------------------------
st.subheader("1) Upload CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        column_info, sample_rows = extract_schema_and_samples(df, sample_rows=5)
        st.session_state.schema = {"columns": list(df.columns)}
        st.session_state.sample_rows = sample_rows
        st.session_state.column_info = column_info

        st.success("File loaded successfully.")
        st.write("Preview (first 10 rows):")
        st.dataframe(df.head(10))

        with st.expander("Schema and dtypes"):
            st.json({"columns": column_info})
        with st.expander("Sample rows used for context"):
            st.json(sample_rows)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")


# ------------------------
# Natural Language Query UI
# ------------------------
st.subheader("2) Ask a question about your data")
question = st.text_input("Ask in natural language, e.g., 'Which customers had the highest downtime?'", "")
show_code = st.toggle("Show generated code", value=True)
max_retries = 3

run_button = st.button("Run")

if run_button:
    if st.session_state.df is None:
        st.warning("Please upload a CSV first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        df = st.session_state.df
        column_info = st.session_state.column_info or {}
        sample_rows = st.session_state.sample_rows or []

        # Build and send initial prompt
        prompt = build_prompt(question.strip(), column_info, sample_rows)
        llm_response = query_llm(prompt, endpoint=endpoint, model=model, temperature=temperature)

        if llm_response.startswith("ERROR_CALLING_LLM:"):
            st.error(llm_response)
        else:
            code = extract_code_block(llm_response)
            if not code:
                st.error("Could not extract a Python code block from the LLM response.")
            else:
                st.session_state.last_code = code
                attempt = 0
                success = False
                last_error = ""
                last_env: Dict[str, Any] = {}

                while attempt < max_retries and not success:
                    attempt += 1
                    with st.spinner(f"Executing generated code (attempt {attempt}/{max_retries})..."):
                        ok, output, env_after = execute_user_code(code, df)
                        last_env = env_after or {}
                        if ok:
                            success = True
                            if output:
                                with st.expander("Program output (stdout/stderr)"):
                                    st.code(output)
                        else:
                            last_error = output
                            repair_prompt = (
                                "Your previous code raised an exception. Here is the code and the error. "
                                "Return corrected Python code only in one fenced code block. Do not explain.\n\n"
                                f"Previous code:\n```python\n{code}\n```\n\n"
                                f"Error traceback:\n```\n{last_error}\n```\n"
                            )
                            llm_repair = query_llm(repair_prompt, endpoint=endpoint, model=model, temperature=temperature)
                            new_code = extract_code_block(llm_repair)
                            if new_code:
                                code = new_code
                                st.session_state.last_code = code
                            else:
                                break

                table_obj = None
                if success:
                    # Try to locate a DataFrame/Series result to ensure visualization even if LLM forgot to use st.*
                    preferred_keys = ["result", "results", "out", "output", "table", "data"]
                    for key in preferred_keys:
                        obj = last_env.get(key)
                        if isinstance(obj, (pd.DataFrame, pd.Series)):
                            table_obj = obj
                            break
                    if table_obj is None:
                        for key, obj in list(last_env.items()):
                            if key == "df":
                                continue
                            if isinstance(obj, (pd.DataFrame, pd.Series)):
                                table_obj = obj
                                break

                    if table_obj is not None:
                        if isinstance(table_obj, pd.Series):
                            table_df = table_obj.to_frame(name=table_obj.name or "value")
                        else:
                            table_df = table_obj
                        st.subheader("Visualization Result")
                        st.dataframe(table_df.head(50))
                        # Simple auto bar chart if small and mostly numeric
                        try:
                            if table_df.shape[0] <= 50 and table_df.select_dtypes(include=[np.number]).shape[1] >= 1:
                                st.bar_chart(table_df.select_dtypes(include=[np.number]).head(50))
                        except Exception:
                            pass
                else:
                    st.error("Failed to execute generated code after retries.")
                    with st.expander("Last error"):
                        st.code(last_error)

                # Always produce insights: use table result if available, else fall back to CSV preview
                try:
                    if success and table_obj is not None:
                        if isinstance(table_obj, pd.Series):
                            table_df = table_obj.to_frame(name=table_obj.name or "value")
                        else:
                            table_df = table_obj
                        preview_rows = min(10, len(table_df))
                        table_preview = table_df.head(preview_rows)
                        try:
                            preview_json = table_preview.reset_index().to_json(orient="records")
                        except Exception:
                            preview_json = json.dumps([], ensure_ascii=False)
                    else:
                        # Fallback to CSV preview
                        fallback_df = df.head(10)
                        try:
                            preview_json = fallback_df.reset_index().to_json(orient="records")
                        except Exception:
                            preview_json = json.dumps([], ensure_ascii=False)

                    insights_prompt = (
                        "Based on the user's question and the tabular preview, write concise insights, likely causes, "
                        "and suggested next steps in markdown. Do not return code.\n\n"
                        f"User question: {question}\n\n"
                        f"Schema (column: dtype):\n" + "\n".join([f"- {c}: {t}" for c, t in (column_info or {}).items()]) + "\n\n"
                        f"Tabular preview (JSON, with index if present):\n{preview_json}\n\n"
                        "Structure with short sections: Insights, Possible Causes, Suggested Actions."
                    )
                    insights_text = query_llm_text(
                        insights_prompt,
                        endpoint=endpoint,
                        model=model,
                        temperature=max(0.2, float(temperature)),
                        max_tokens=800,
                    )
                    st.subheader("Insights")
                    st.markdown(insights_text)
                except Exception:
                    pass

                if show_code and st.session_state.last_code:
                    with st.expander("Generated code"):
                        st.code(st.session_state.last_code, language="python")


# Footer note
st.caption("Powered by LM Studio (OpenAI-compatible local API). Your data stays local.")


