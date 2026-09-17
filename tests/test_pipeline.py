"""Runnable with plain `python tests/test_pipeline.py` (no pytest needed)."""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentcore.loop import Agent
from agentcore.memory import Memory
from agentcore.guardrails import Guardrails
from mlops.model import LogisticRegression
from mlops.data import make_batch, feature_means
from mlops.controller import (ModelRegistry, ControllerBrain, build_registry_tools,
                              retrain, promote)


def make_reg():
    Xtr, ytr = make_batch(n=500, drift=0.0, seed=1)
    return ModelRegistry(LogisticRegression().fit(Xtr, ytr), feature_means(Xtr))


def run_episode(reg, batch, holdout):
    agent = Agent(brain=ControllerBrain(), tools=build_registry_tools(),
                  memory=Memory(), guardrails=Guardrails())
    return agent.run("keep the served model healthy",
                     {"registry": reg, "batch": batch, "eval": holdout})


def test_heals_after_drift():
    reg = make_reg()
    batch = make_batch(n=300, drift=2.0, seed=20)
    holdout = make_batch(n=300, drift=2.0, seed=21)
    before = reg.current.accuracy(*batch)
    state = run_episode(reg, batch, holdout)
    after = reg.current.accuracy(*batch)
    assert before < 0.75, "drift should degrade accuracy, got %.2f" % before
    assert after > 0.90, "controller should heal accuracy, got %.2f" % after
    assert state["signals"]["action"] == "promoted"
    print("PASS: heals after drift  (%.2f -> %.2f, v%d)" % (before, after, reg.version))


def test_guardrail_blocks_worse_candidate():
    reg = make_reg()
    ver0, model0 = reg.version, reg.current
    r = random.Random(0)
    Xg = [[r.gauss(0, 1), r.gauss(0, 1)] for _ in range(300)]
    yg = [r.randint(0, 1) for _ in range(300)]        # garbage labels -> weak candidate
    holdout = make_batch(n=300, drift=0.0, seed=30)    # clean eval the current model aces
    state = {"context": {"registry": reg, "batch": (Xg, yg), "eval": holdout},
             "artifacts": {}, "signals": {}}
    retrain(state)
    res = promote(state)
    assert reg.version == ver0 and reg.current is model0, "a worse model must NOT be promoted"
    assert reg.blocked == 1 and "kept_current" in res["action"]
    print("PASS: guardrail blocked a worse candidate  (%s)" % res)


def test_healthy_batch_only_monitors():
    reg = make_reg()
    batch = make_batch(n=300, drift=0.0, seed=40)
    holdout = make_batch(n=300, drift=0.0, seed=41)
    state = run_episode(reg, batch, holdout)
    assert state["signals"]["action"] == "monitor"
    assert reg.promotions == 0
    print("PASS: healthy batch only monitors (no needless retrain)")



def test_real_elec2_pipeline():
    from mlops.pipeline import build_real_steps, run_pipeline
    init, steps, thr = build_real_steps()
    assert len(steps) >= 5, "expected several streamed windows of real data"
    res = run_pipeline(init, steps, thr)
    assert res["promotions"] >= 1, "should heal at least once on real drift"
    assert res["final_version"] >= 2
    assert len(res["rows"]) == len(steps)
    print("PASS: real ELEC2 pipeline (promotions=%d, blocked=%d, worst=%.2f)"
          % (res["promotions"], res["blocked"], res["worst"]))


if __name__ == "__main__":
    test_heals_after_drift()
    test_guardrail_blocks_worse_candidate()
    test_healthy_batch_only_monitors()
    test_real_elec2_pipeline()
    print("\nAll tests passed.")
