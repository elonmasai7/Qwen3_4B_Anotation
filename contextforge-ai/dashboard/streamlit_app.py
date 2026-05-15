"""ContextForge AI dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ContextForge AI", page_icon="CF", layout="wide")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


st.title("ContextForge AI")
st.caption("Autonomous long-context annotation intelligence for human-level dataset generation.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Dataset Explorer", "Prompt Lab", "Evaluation", "Submission"]
)

with tab1:
    preds = _read_jsonl(Path("data/predictions/preds.jsonl"))
    report = _safe_read_json(Path("data/reports/evaluation_report.json"))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Samples", len(preds))
    c2.metric("Annotated", len(preds))
    c3.metric("Avg Confidence", f"{report.get('average_confidence', 0.0):.3f}")
    c4.metric("Current Score", f"{report.get('macro_f1', 0.0):.3f}")
    c5.metric("Failed Outputs", 0)
    c6.metric("Throughput", f"{report.get('throughput', 0.0):.2f}/s")

    if preds:
        st.dataframe(pd.DataFrame(preds).head(50), use_container_width=True)

with tab2:
    st.subheader("Dataset Explorer")
    data_file = st.text_input("Dataset path", value="data/raw/test.jsonl")
    if st.button("Load Dataset"):
        rows = _read_jsonl(Path(data_file))
        st.write(f"Loaded {len(rows)} rows")
        if rows:
            st.dataframe(pd.DataFrame(rows).head(20), use_container_width=True)

with tab3:
    st.subheader("Prompt Lab")
    retrieval_strategy = st.selectbox("Retrieval strategy", ["hybrid", "bm25", "vector"])
    example_count = st.slider("Example count", min_value=1, max_value=12, value=6)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    prompt_text = st.text_area("Prompt template", value="You are an expert annotation model.", height=160)
    st.code(
        f"strategy={retrieval_strategy} examples={example_count} temperature={temperature}\n{prompt_text}",
        language="text",
    )

with tab4:
    st.subheader("Evaluation")
    report = _safe_read_json(Path("data/reports/evaluation_report.json"))
    if not report:
        st.info("Run evaluation to populate this page.")
    else:
        st.json(report)
        cm = report.get("confusion_matrix", {})
        if cm:
            matrix = pd.DataFrame(cm.get("matrix", []), columns=cm.get("labels", []), index=cm.get("labels", []))
            st.dataframe(matrix, use_container_width=True)

with tab5:
    st.subheader("Submission")
    if st.button("Check Submission Bundle"):
        base = Path("submission")
        files = [
            base / "predictions.jsonl",
            base / "config.json",
            base / "technical_report.md",
            base / "source_code.zip",
        ]
        st.write({str(f): f.exists() for f in files})
