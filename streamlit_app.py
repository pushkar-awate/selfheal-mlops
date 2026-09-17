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
    "**Machine-learning models silently get worse as the real world changes.** "
    "This is an autonomous agent that watches a live model and keeps it accurate: "
    "on each new batch of data it checks whether the model is slipping, retrains a "
    "fresh one when it is, and **ships the new model only if it is actually better "
    "than the old one** - so a bad model never reaches production. Every decision "
    "is logged."
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
    st.line_chart({"model left alone (no self-healing)": res["frozen_acc"],
                   "model with self-healing (this agent)": res["healed_acc"]})

    lift = round((res["healed_acc"][-1] - res["frozen_acc"][-1]) * 100)
    st.markdown(
        "**Read the chart:** the lower line is a model left alone - it decays as "
        "the data drifts. The upper line is the same model kept healthy by this "
        "agent. By the end it is about **%d points more accurate**, with no human "
        "in the loop." % lift)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Better models shipped", res["promotions"])
    m2.metric("Bad models caught", res["blocked"])
    m3.metric("Lowest accuracy seen", "%d%%" % round(res["worst"] * 100))
    m4.metric("Model version now live", "v%d" % res["final_version"])

    if res["blocked"] > 0:
        st.success(
            "✓ The agent caught %d retrained model(s) that were NOT actually "
            "better and refused to ship them - a worse model never reached "
            "production. Knowing when NOT to deploy is the hard part of MLOps." %
            res["blocked"])

    st.subheader("What the agent decided, %s by %s" % (xlabel, xlabel))
    st.table(res["rows"])
    st.caption("Each row is one batch of data. The agent monitors while the model "
               "is healthy, retrains when it slips, and only swaps in the new "
               "model when it wins on unseen data - otherwise it keeps the old one.")

st.divider()
st.caption("Real data: ELEC2 (NSW electricity market, a standard concept-drift "
           "benchmark). Built on a from-scratch agent runtime (agentcore). "
           "Source: github.com/pushkar-awate/selfheal-mlops")
