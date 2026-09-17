"""Live web demo of the self-healing ML pipeline. Deploy on Streamlit Cloud.

A visitor picks a drift level and watches, in real time, the model's accuracy
collapse as data drifts and recover as the controller retrains and safely
promotes better models. No API key, no setup - pure Python.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentcore.loop import Agent
from agentcore.memory import Memory
from agentcore.guardrails import Guardrails
from mlops.model import LogisticRegression
from mlops.data import make_batch, feature_means
from mlops.controller import ModelRegistry, ControllerBrain, build_registry_tools

st.set_page_config(page_title="Self-Healing MLOps", page_icon="\U0001FA79", layout="centered")

st.title("\U0001FA79 Self-Healing ML Pipeline")
st.markdown(
    "Machine-learning models quietly rot as real-world data drifts. This is an "
    "**autonomous agent that keeps a live model healthy**: every batch it checks "
    "for drift and accuracy loss, retrains when the model degrades, and **promotes "
    "a new model only if it beats the current one** (a guardrail that blocks bad "
    "deploys). Pick a drift level and press run."
)

col1, col2 = st.columns(2)
peak_drift = col1.slider("Peak data drift", 0.5, 4.0, 3.0, 0.5,
                         help="How far the incoming data shifts from what the model was trained on.")
n_batches = col2.slider("Batches to stream", 4, 12, 8)

if st.button("Run the self-healing simulation", type="primary"):
    schedule = [round(peak_drift * i / (n_batches - 1), 2) for i in range(n_batches)]

    Xtr, ytr = make_batch(n=500, drift=0.0, seed=1)
    frozen = LogisticRegression().fit(Xtr, ytr)                 # never retrained (control)
    reg = ModelRegistry(LogisticRegression().fit(Xtr, ytr), feature_means(Xtr))

    frozen_acc, healed_acc, rows = [], [], []
    worst = 1.0
    progress = st.progress(0.0)
    for i, d in enumerate(schedule):
        batch = make_batch(n=300, drift=d, seed=100 + i)
        holdout = make_batch(n=300, drift=d, seed=500 + i)
        fa = frozen.accuracy(*batch)
        before = reg.current.accuracy(*batch)
        worst = min(worst, before)
        agent = Agent(brain=ControllerBrain(), tools=build_registry_tools(),
                      memory=Memory(), guardrails=Guardrails())
        state = agent.run("keep the served model healthy",
                          {"registry": reg, "batch": batch, "eval": holdout})
        after = reg.current.accuracy(*batch)
        frozen_acc.append(round(fa, 3))
        healed_acc.append(round(after, 3))
        rows.append({"batch": i, "drift": d, "accuracy before": round(before, 2),
                     "action": state["signals"].get("action", "-"),
                     "accuracy after": round(after, 2), "model": "v%d" % reg.version})
        progress.progress((i + 1) / len(schedule))

    st.subheader("Accuracy over time")
    st.line_chart({"no self-healing (frozen model)": frozen_acc,
                   "with self-healing (this controller)": healed_acc})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Models promoted", reg.promotions)
    m2.metric("Bad deploys blocked", reg.blocked)
    m3.metric("Worst accuracy hit", "%d%%" % round(worst * 100))
    m4.metric("Final model", "v%d" % reg.version)

    st.subheader("What the controller decided, batch by batch")
    st.table(rows)
    st.caption("The frozen model degrades as data drifts; the self-healing "
               "controller detects each collapse, retrains, and safely promotes a "
               "better model - recovering accuracy with zero bad deploys.")

st.divider()
st.caption("Built on a from-scratch agent runtime (agentcore). "
           "Source: github.com/pushkar-awate/selfheal-mlops")
