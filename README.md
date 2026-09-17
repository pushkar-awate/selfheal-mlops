# selfheal-mlops

[![tests](https://github.com/pushkar-awate/selfheal-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/pushkar-awate/selfheal-mlops/actions/workflows/ci.yml)

An autonomous controller that keeps a machine-learning model healthy under data
drift. On each window of data it measures drift and accuracy, and when the model
degrades it retrains, evaluates a candidate on held-out data, and **promotes it
only if it beats the current model** - otherwise it keeps the old one. Every
decision is logged.

By default it runs on **real data**: the ELEC2 electricity-market dataset, a
widely-used concept-drift benchmark. A synthetic "stress test" mode lets you dial
drift up on demand.

It is built on the same `agentcore` runtime as
[jobfit-agent](https://github.com/pushkar-awate/jobfit-agent) - the same
perceive -> reason -> guardrail -> act -> verify -> remember loop, a different
brain and tool set. Two working apps on one core.

## Live demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://selfheal-mlops.streamlit.app)

**Try it live (no install): https://selfheal-mlops.streamlit.app**

![Self-healing pipeline demo](assets/selfheal-demo.gif)

*One click on real ELEC2 data: the model left alone (lower line) decays as the market drifts, while the self-healed model (upper line) stays accurate - 2 better models shipped, 2 bad retrains caught before shipping.*

Stream the real electricity data (or the synthetic stress test) and watch the
controller detect drift, retrain, promote better models, and block worse ones.

## What it does (real output, ELEC2)

```
$ python -m mlops.run
data source: real ELEC2 electricity-market data

window     acc_before  action           acc_after  model
window 3   0.78        promoted         0.76       v2
window 8   0.78        kept_current     0.78       v2   <- guardrail blocked a non-improvement
window 10  0.73        promoted         0.74       v3
window 11  0.78        kept_current     0.78       v3   <- guardrail blocked again
promotions: 2 | guardrail-blocked: 2 | worst accuracy: 0.73 | final: v3
```

On real market data the controller heals real accuracy dips **and** refuses to
ship retrains that would not improve the model - the safety gate that is the hard
part of MLOps.

## How it works

Each window of data is one agent episode on `agentcore`:

1. **Perceive / assess_health** - measure feature drift (vs the current serving
   distribution) and the live model's accuracy.
2. **Reason** - the `ControllerBrain` policy: heal if drift or accuracy breaches
   thresholds, else monitor.
3. **Act** - retrain a candidate on the freshly labeled window.
4. **Verify / guardrail** - promote the candidate **only if** it beats the
   current model on the next (held-out) window; otherwise keep the current model
   and record that the guardrail blocked a non-improvement.
5. **Remember** - append the decision to `decisions.jsonl`.

## Modes

- **Real (default):** streams the ELEC2 electricity-market dataset chronologically;
  drift is genuine and documented.
- **Synthetic stress test** (`python -m mlops.run --synthetic`): generates data
  with controllable drift so you can trigger a dramatic heal on demand.

## Data

ELEC2 ("electricity") is a standard real-world concept-drift benchmark: 45,312
half-hourly records from the New South Wales electricity market, features
normalized to 0-1, label = price movement UP/DOWN. It is bundled in
`mlops/data_files/elec2.csv` so the app runs with no external dependency.
Source: M. Harries, "Splice-2 Comparative Evaluation: Electricity Pricing" (1999);
distributed via scikit-multiflow.

## What this is (and isn't)

- **Real data, replayed** - not a live feed. Replaying a real drift dataset means
  the drift is authentic *and* the demo reliably shows healing every run (a live
  API might not drift when someone clicks). In production the same loop would run
  on live traffic; `docs/scaling-to-k8s.md` maps each part to MLflow, Prometheus,
  Evidently, a Kubernetes CronJob, and Terraform.
- **The control logic is production-real** - drift detection, retrain, held-out
  evaluation, guardrail-gated promotion, rollback, and an audit ledger.

## Future work

- **Live data source.** A "live" mode could fetch a fresh window from a public
  API (for example Coinbase crypto candles) on each run instead of replaying
  ELEC2. ELEC2 stays the default on purpose: a live feed may not drift on the
  day someone clicks, and the demo's value is that it reliably shows the full
  detect -> retrain -> promote/block story every time. The data loader is
  already isolated, so adding a source is a drop-in.
- **Bigger labeled evaluation** and wiring the decision ledger into a live
  monitoring dashboard.

## Project layout

```
agentcore/   reusable core, shared with jobfit-agent (loop, brain, tools, memory, guardrails)
mlops/       model.py, data.py (synthetic), data_real.py (ELEC2 loader),
             controller.py, pipeline.py (shared runner), run.py, data_files/elec2.csv
tests/       heal, guardrail-block, healthy-monitor, and real-ELEC2 tests
docs/        scaling-to-k8s.md
streamlit_app.py   the live web UI
```

## Quickstart

```
git clone https://github.com/pushkar-awate/selfheal-mlops.git
cd selfheal-mlops
python -m mlops.run              # real ELEC2 data
python -m mlops.run --synthetic  # controllable synthetic drift
python tests/test_pipeline.py    # tests
```

Core is pure standard library; the web UI uses Streamlit. Requires Python 3.8+.
