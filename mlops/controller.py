"""The self-healing controller: a ControllerBrain + tools on agentcore.

Per incoming batch the agent runs one episode:
  assess_health -> (retrain -> promote) if unhealthy, else monitor -> finish
The promote step carries the key guardrail: a candidate is only promoted if it
beats the current model on a held-out set, so a bad retrain can never degrade
production. Every decision lands in the ledger.
"""
from __future__ import annotations

from agentcore.loop import Decision
from agentcore.brain import Brain
from agentcore.tools import ToolRegistry
from mlops.model import LogisticRegression
from mlops.data import drift_score, feature_means

DRIFT_THRESHOLD = 0.5   # mean feature shift that counts as drift
ACC_FLOOR = 0.80        # live accuracy below this is unhealthy
PROMOTE_MARGIN = 0.01   # candidate must beat current by at least this


class ModelRegistry:
    def __init__(self, model, baseline_means, version=1):
        self.current = model
        self.baseline_means = baseline_means
        self.version = version
        self.history = []
        self.promotions = 0
        self.rollbacks = 0
        self.blocked = 0

    def promote(self, model):
        self.history.append((self.version, self.current))
        self.version += 1
        self.current = model

    def rollback(self):
        if self.history:
            v, m = self.history.pop()
            self.current = m
            self.version = v
            self.rollbacks += 1


class ControllerBrain(Brain):
    """Deterministic control policy that branches on observed signals."""

    def decide(self, observation, tool_specs):
        done = observation.get("completed", [])
        sig = observation.get("signals", {})
        if "assess_health" not in done:
            return Decision("assess_health", {}, "check drift and live accuracy")
        drift, acc = sig.get("drift", 0.0), sig.get("acc", 1.0)
        unhealthy = drift > DRIFT_THRESHOLD or acc < ACC_FLOOR
        if unhealthy:
            if "retrain" not in done:
                return Decision("retrain", {},
                                "drift %.2f / acc %.2f breached thresholds -> retrain"
                                % (drift, acc))
            if "promote" not in done:
                return Decision("promote", {}, "evaluate candidate; promote only if better")
            return Decision("finish", {}, "healed")
        if "monitor" not in done:
            return Decision("monitor", {}, "healthy: drift %.2f / acc %.2f" % (drift, acc))
        return Decision("finish", {}, "healthy")

    def generate(self, task, payload):
        return ""


# ---- tools (operate on state["context"]) --------------------------------

def assess_health(state, **kw):
    ctx = state["context"]
    X, y = ctx["batch"]
    reg = ctx["registry"]
    drift = drift_score(X, reg.baseline_means)
    acc = reg.current.accuracy(X, y)
    state["signals"]["drift"] = drift
    state["signals"]["acc"] = acc
    state["artifacts"]["health"] = {"drift": round(drift, 3), "acc": round(acc, 3)}
    return state["artifacts"]["health"]


def retrain(state, **kw):
    ctx = state["context"]
    X, y = ctx["batch"]
    candidate = LogisticRegression().fit(X, y)
    Xe, ye = ctx["eval"]
    cand_acc = candidate.accuracy(Xe, ye)
    state["artifacts"]["candidate"] = {"model": candidate, "eval_acc": cand_acc}
    state["signals"]["cand_acc"] = cand_acc
    return {"cand_eval_acc": round(cand_acc, 3)}


def promote(state, **kw):
    ctx = state["context"]
    reg = ctx["registry"]
    Xe, ye = ctx["eval"]
    X_batch = ctx["batch"][0]
    cand = state["artifacts"]["candidate"]["model"]
    cur_acc = reg.current.accuracy(Xe, ye)
    cand_acc = cand.accuracy(Xe, ye)
    if cand_acc >= cur_acc + PROMOTE_MARGIN:          # <-- safety guardrail
        reg.promote(cand)
        reg.promotions += 1
        reg.baseline_means = feature_means(X_batch)   # drift now measured vs the new normal
        state["signals"]["action"] = "promoted"
        result = {"action": "promoted", "from": round(cur_acc, 3),
                  "to": round(cand_acc, 3), "version": reg.version}
    else:
        reg.blocked += 1                              # guardrail blocked a non-improvement
        state["signals"]["action"] = "kept_current"
        result = {"action": "kept_current (guardrail)", "current": round(cur_acc, 3),
                  "candidate": round(cand_acc, 3)}
    state["artifacts"]["promotion"] = result
    return result


def monitor(state, **kw):
    state["signals"]["action"] = "monitor"
    return {"action": "monitor"}


def build_registry_tools():
    reg = ToolRegistry()
    reg.register("assess_health", assess_health,
                 "Measure feature drift and the live model's accuracy on the batch.")
    reg.register("retrain", retrain,
                 "Train a candidate model on the freshly labeled batch.")
    reg.register("promote", promote,
                 "Promote the candidate only if it beats the current model (guardrail).")
    reg.register("monitor", monitor, "Record a healthy check; make no change.")
    return reg
