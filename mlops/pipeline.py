"""Shared runner: build a list of streamed steps (real ELEC2 or synthetic) and
run the self-healing controller over them."""
from __future__ import annotations

from agentcore.loop import Agent
from agentcore.memory import Memory
from agentcore.guardrails import Guardrails
from mlops.model import LogisticRegression
from mlops.data import make_batch, feature_means
from mlops.data_real import load_elec2
from mlops.controller import ModelRegistry, ControllerBrain, build_registry_tools


def _xy(rows):
    return [x for x, _ in rows], [y for _, y in rows]


def build_real_steps(train_n=3000, batch_size=1000, max_batches=12):
    data = load_elec2()
    chunks = []
    idx = train_n
    while idx < len(data) and len(chunks) < max_batches + 1:
        chunks.append(data[idx:idx + batch_size])
        idx += batch_size
    steps = []
    for i in range(len(chunks) - 1):
        steps.append({"batch": _xy(chunks[i]), "eval": _xy(chunks[i + 1]),
                      "label": "window %d" % (i + 1)})
    return _xy(data[:train_n]), steps, {"drift_threshold": 0.08, "acc_floor": 0.78}


def build_synthetic_steps(peak_drift=3.0, n_batches=8):
    sched = [round(peak_drift * i / (n_batches - 1), 2) for i in range(n_batches)]
    steps = []
    for i, d in enumerate(sched):
        steps.append({"batch": make_batch(300, d, 100 + i),
                      "eval": make_batch(300, d, 500 + i), "label": "drift %.1f" % d})
    return make_batch(500, 0.0, 1), steps, {"drift_threshold": 0.5, "acc_floor": 0.80}


def run_pipeline(init_train, steps, thresholds):
    Xtr, ytr = init_train
    frozen = LogisticRegression().fit(Xtr, ytr)                 # never retrained (control)
    reg = ModelRegistry(LogisticRegression().fit(Xtr, ytr), feature_means(Xtr))
    rows, frozen_acc, healed_acc, worst = [], [], [], 1.0
    for i, step in enumerate(steps):
        batch, ev = step["batch"], step["eval"]
        fa = frozen.accuracy(*batch)
        before = reg.current.accuracy(*batch)
        worst = min(worst, before)
        agent = Agent(brain=ControllerBrain(**thresholds), tools=build_registry_tools(),
                      memory=Memory(), guardrails=Guardrails())
        state = agent.run("keep the served model healthy",
                          {"registry": reg, "batch": batch, "eval": ev})
        after = reg.current.accuracy(*batch)
        frozen_acc.append(round(fa, 3))
        healed_acc.append(round(after, 3))
        rows.append({"window": step["label"], "accuracy before": round(before, 2),
                     "action": state["signals"].get("action", "-"),
                     "accuracy after": round(after, 2), "model": "v%d" % reg.version})
    return {"rows": rows, "frozen_acc": frozen_acc, "healed_acc": healed_acc,
            "promotions": reg.promotions, "blocked": reg.blocked,
            "worst": worst, "final_version": reg.version}
