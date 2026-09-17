# selfheal-mlops

An autonomous controller that keeps a machine-learning model healthy under data
drift. Every batch it measures drift and live accuracy, and when the model
degrades it retrains, evaluates a candidate, and **promotes it only if it beats
the current model** - otherwise it keeps the old one. Every decision is logged.

It is built on the same `agentcore` runtime as
[jobfit-agent](https://github.com/pushkar-awate/jobfit-agent): the same
perceive -> reason -> guardrail -> act -> verify -> remember loop, a different
brain and tool set. Two working apps on one core.


## Live demo

**Try it in your browser (no install): _deploying — link coming here_**

Pick a drift level and watch the model's accuracy crash and then recover as the controller heals it, live.

The agent + pipeline core is pure standard library (the only dependency, Streamlit, is just for the web UI). Clone and run:

```
$ python -m mlops.run
#   drift  acc_before  action        acc_after  ver
2   1.0    0.59        promoted      0.99       v2
4   2.0    0.57        promoted      0.99       v3
6   3.0    0.58        promoted      0.97       v4
promotions: 3 | guardrail-blocked: 0 | worst live accuracy before healing: 0.57
```

The model's accuracy collapses to ~57% each time the data drifts; the controller
detects it, retrains on the fresh batch, safely promotes the better model, and
recovers to ~97-99% - with zero bad promotions.

## How it works

Each incoming batch is one agent episode on `agentcore`:

1. **Perceive / assess_health** - compute feature drift (vs the current serving
   distribution) and the live model's accuracy on the batch.
2. **Reason** - the `ControllerBrain` policy: if drift or accuracy breaches its
   thresholds, heal; otherwise just monitor.
3. **Act** - `retrain` a candidate on the freshly labeled batch.
4. **Verify / guardrail** - `promote` the candidate **only if** it beats the
   current model on a held-out set by a margin; otherwise keep the current model
   and record that the guardrail blocked a non-improvement.
5. **Remember** - append the decision, rationale and result to `decisions.jsonl`.

After a promotion the drift baseline resets to the new serving distribution, so
drift is always measured against "now" - the way real monitoring works.

### The guardrail is real

A retrain that produces a *worse* model is never shipped:

```
$ python tests/test_pipeline.py
PASS: heals after drift  (0.58 -> 1.00, v2)
PASS: guardrail blocked a worse candidate  (current 0.997 vs candidate 0.517 -> kept_current)
PASS: healthy batch only monitors (no needless retrain)
```

## Project layout

```
agentcore/   the reusable core, shared with jobfit-agent (extended here with a
             small 'signals' scratchpad so the brain can branch on observed metrics)
mlops/       model.py (pure-Python logistic regression), data.py (drifting data),
             controller.py (ControllerBrain + tools + model registry), run.py
tests/       heals-after-drift, guardrail-blocks-worse-model, healthy-only-monitors
docs/        scaling-to-k8s.md - how this maps to a production Kubernetes stack
```

## Limitations

Honest boundaries, by design:

- **Synthetic data.** The drift is generated, not real; it exists to exercise the
  control loop deterministically. The logic is what transfers, not the numbers.
- **In-process, not distributed.** The model registry, serving and controller run
  in one Python process. `docs/scaling-to-k8s.md` shows how each piece maps to a
  real Kubernetes + Terraform + MLflow + Prometheus/Grafana stack; this repo runs
  the *logic* so it stays cloneable in seconds without a cluster.
- **Labels assumed available.** Healing retrains on freshly labeled batches. In
  production, labels lag, which is exactly why drift is the leading indicator.

## Quickstart

```
git clone https://github.com/pushkar-awate/selfheal-mlops.git
cd selfheal-mlops
python -m mlops.run       # watch the self-healing log
python tests/test_pipeline.py
```

Requires Python 3.8+. No pip install needed.
