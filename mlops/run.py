"""Run the self-healing controller over a data stream.

  python -m mlops.run                 # real ELEC2 electricity-market data (default)
  python -m mlops.run --synthetic     # synthetic stress-test stream (controllable drift)
"""
from __future__ import annotations
import argparse

from mlops.pipeline import build_real_steps, build_synthetic_steps, run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic stress-test stream instead of real ELEC2")
    ap.add_argument("--drift", type=float, default=3.0, help="peak drift (synthetic only)")
    ap.add_argument("--batches", type=int, default=8, help="batches (synthetic only)")
    args = ap.parse_args()

    if args.synthetic:
        init, steps, thr = build_synthetic_steps(args.drift, args.batches)
        src = "synthetic stress test"
    else:
        init, steps, thr = build_real_steps()
        src = "real ELEC2 electricity-market data"

    res = run_pipeline(init, steps, thr)
    print("data source: %s\n" % src)
    print("%-10s %-11s %-24s %-10s %s"
          % ("window", "acc_before", "action", "acc_after", "model"))
    print("-" * 68)
    for r in res["rows"]:
        print("%-10s %-11s %-24s %-10s %s"
              % (r["window"], r["accuracy before"], r["action"],
                 r["accuracy after"], r["model"]))
    print("-" * 68)
    print("promotions: %d | guardrail-blocked: %d | worst accuracy: %.2f | final: v%d"
          % (res["promotions"], res["blocked"], res["worst"], res["final_version"]))


if __name__ == "__main__":
    main()
