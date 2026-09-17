"""Stream drifting batches through the self-healing controller.

  python -m mlops.run
"""
from __future__ import annotations
import os

from agentcore.loop import Agent
from agentcore.memory import Memory
from agentcore.guardrails import Guardrails
from mlops.model import LogisticRegression
from mlops.data import make_batch, feature_means
from mlops.controller import ModelRegistry, ControllerBrain, build_registry_tools

DRIFT_SCHEDULE = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def main(ledger="decisions.jsonl"):
    # train and deploy the initial model at drift 0
    Xtr, ytr = make_batch(n=500, drift=0.0, seed=1)
    model = LogisticRegression().fit(Xtr, ytr)
    reg = ModelRegistry(model, baseline_means=feature_means(Xtr))

    if os.path.exists(ledger):
        os.remove(ledger)

    print("%-3s %-6s %-11s %-26s %-10s %s"
          % ("#", "drift", "acc_before", "action", "acc_after", "ver"))
    print("-" * 72)

    worst = 1.0
    for i, d in enumerate(DRIFT_SCHEDULE):
        batch = make_batch(n=300, drift=d, seed=100 + i)
        holdout = make_batch(n=300, drift=d, seed=500 + i)
        acc_before = reg.current.accuracy(*batch)
        worst = min(worst, acc_before)

        agent = Agent(brain=ControllerBrain(), tools=build_registry_tools(),
                      memory=Memory(path=ledger), guardrails=Guardrails())
        state = agent.run(goal="keep the served model healthy",
                          context={"registry": reg, "batch": batch, "eval": holdout})

        acc_after = reg.current.accuracy(*batch)
        action = state["signals"].get("action", "-")
        print("%-3d %-6.1f %-11.2f %-26s %-10.2f v%d"
              % (i, d, acc_before, action, acc_after, reg.version))

    print("-" * 72)
    print("promotions: %d | guardrail-blocked: %d | rollbacks: %d"
          % (reg.promotions, reg.blocked, reg.rollbacks))
    print("worst live accuracy seen before healing: %.2f" % worst)
    print("final model version: v%d   (ledger: %s)" % (reg.version, ledger))


if __name__ == "__main__":
    main()
