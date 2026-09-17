"""Load the real ELEC2 electricity-market dataset (a standard concept-drift
benchmark). Features are already normalized 0-1; label 'class' is UP(1)/DOWN(0).
Source: Harries (1999), NSW electricity market; via scikit-multiflow.
"""
from __future__ import annotations
import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ELEC2_PATH = os.path.join(_HERE, "data_files", "elec2.csv")
FEATURES = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]


def load_elec2(path=ELEC2_PATH):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                x = [float(row[c]) for c in FEATURES]
                y = int(float(row["class"]))
            except (KeyError, ValueError):
                continue
            rows.append((x, y))
    return rows
