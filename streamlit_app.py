"""Live web demo of the self-healing ML pipeline (real ELEC2 data by default).

A visitor streams real electricity-market data (a standard concept-drift
benchmark) through an autonomous controller and watches it detect drift,
retrain, and either promote a better model or block a worse one - live.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlops.pipeline import build_real_steps, build_synthetic_steps, run_pipeline

st.set_page_config(page_title="Self-Healing MLOps", page_icon="\U0001FA79", layout="centered")

st.title("\U0001FA79 Self-Healing ML Pipeline")
st.markdown(
    "Machine-learning models quietly rot as real-world data drifts. This is an "
    "**autonomous agent that keeps a live model healthy**: on each window of data "
    "it checks for drift and accuracy loss, retrains when the model degrades, and "
    "**promotes a new model only if it beats the current one** - blocking bad "
    "deploys. Every decision is logged."
)

source = st.radio(
    "Data source",
    ["Real electricity-market data (ELEC2)", "Synthetic stress test"],
    help="ELEC2 is a real, widely-used concept-drift benchmark (NSW electricity "
         "market). Synthetic mode lets you dial drift up on demand.",
)

if source.startswith("Synthetic"):
    c1, c2 = st.columns(2)
    peak = c1.slider("Peak data drift", 0.5, 4.0, 3.0, 0.5)
    nb = c2.slider("Batches to stream", 4, 12, 8)

if st.button("Run the self-healing simulation", type="primary"):
    with st.spinner("Streaming data through the controller..."):
        if source.startswith("Synthetic"):
            init, steps, thr = build_synthetic_steps(peak, nb)
            xlabel = "batch"
        else:
            init, steps, thr = build_real_steps()
            xlabel = "window"
        res = run_pipeline(init, steps, thr)

    st.subheader("Accuracy over time")
    st.line_chart({"no self-healing (frozen model)": res["frozen_acc"],
                   "with self-healing (this controller)": res["healed_acc"]})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Models promoted", res["promotions"])
    m2.metric("Bad deploys blocked", res["blocked"])
    m3.metric("Worst accuracy hit", "%d%%" % round(res["worst"] * 100))
    m4.metric("Final model", "v%d" % res["final_version"])

    if res["blocked"] > 0:
        st.success(
            "✓ The guardrail blocked %d retrain(s) that would NOT have improved "
            "the model - a worse model was never deployed. That safety gate is the "
            "hard part of MLOps." % res["blocked"])

    st.subheader("What the controller decided, %s by %s" % (xlabel, xlabel))
    st.table(res["rows"])
    st.caption("The frozen model degrades as data drifts; the self-healing "
               "controller retrains and promotes a better model only when it wins "
               "on held-out data, and blocks it otherwise.")

st.divider()
st.caption("Real data: ELEC2 (NSW electricity market, a standard concept-drift "
           "benchmark). Built on a from-scratch agent runtime (agentcore). "
           "Source: github.com/pushkar-awate/selfheal-mlops")
