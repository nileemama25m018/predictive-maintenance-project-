from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apm.agent.agent import MaintenanceAgent
from apm.agent.tools import MaintenanceTools


st.set_page_config(page_title="Maintenance Copilot", layout="wide")
st.title("Agentic Predictive Maintenance Copilot")

default_sensor_data = ROOT / "data/processed/cmapss_FD001.csv"
if not default_sensor_data.exists():
    default_sensor_data = ROOT / "data/processed/demo_sensor_data.csv"

tools = MaintenanceTools(sensor_data_path=str(default_sensor_data))
agent = MaintenanceAgent(tools)

tabs = st.tabs(["Overview", "Machine Monitor", "AI Copilot"])

with tabs[0]:
    st.subheader("Maintenance Priority")
    priority = tools.rank_maintenance_priority(top_n=10)
    if priority:
        st.dataframe(pd.DataFrame(priority), use_container_width=True)
    else:
        st.info("Train the model and generate demo data to show priority ranking.")

with tabs[1]:
    machine_id = st.number_input("Machine ID", min_value=1, max_value=9999, value=1)
    pred = tools.predict_rul(int(machine_id))
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted RUL", pred.get("predicted_rul", "N/A"))
    c2.metric("Risk", pred.get("risk_level", "N/A"))
    c3.metric("Cycle", pred.get("cycle", "N/A"))

    data_path = default_sensor_data
    if data_path.exists():
        df = pd.read_csv(data_path)
        mdf = df[df["engine_id"] == machine_id]
        sensor_options = [c for c in mdf.columns if c.startswith("sensor_")]
        selected = st.multiselect("Sensors", sensor_options, default=sensor_options[:3])
        if selected and not mdf.empty:
            long_df = mdf[["cycle"] + selected].melt("cycle", var_name="sensor", value_name="value")
            st.plotly_chart(px.line(long_df, x="cycle", y="value", color="sensor"), use_container_width=True)

with tabs[2]:
    query = st.text_input("Ask the copilot", value="Why is Machine 1 high risk?")
    if st.button("Ask", type="primary"):
        response = agent.answer(query)
        st.markdown(response.answer)
        with st.expander("Tool calls"):
            st.write(response.tool_calls)
        with st.expander("Evidence"):
            st.json(response.evidence, expanded=False)
