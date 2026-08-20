#!/usr/bin/env python3
"""Compare model classes for the prospective impact estimator under one protocol.

A result resting on one domain and one model class is weak. The domain half is answered
by running Web, Mail and Vulnerability through one pipeline. This script answers the
model-class half: it holds the feature map, the population, and the organization-held-
out splits fixed at exactly the values ``prospective_impact.py`` uses, and swaps only the
regressor. Tree ensembles from three independent implementations are compared against a
linear model and a neural network, so the choice of gradient-boosted trees is supported by
measurement rather than by citation alone.

Where a model exposes an absolute-error objective we select it, matching the L1 objective
the paper reports. Ridge and the MLP optimize squared error because neither offers an L1
option, which is noted in the output and in the paper.

Usage:
  python3 audit/scripts/model_class_comparison.py --domain web
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain_reliability import DOMAINS, ROOT  # noqa: E402
from prospective_impact import build, mae  # noqa: E402

OUT = ROOT / "outputs" / "model_class_comparison"


def models(seed: int) -> dict:
    """One entry per model class, keyed by the label used in the paper's table."""
    return {
        "LightGBM (reported)": LGBMRegressor(
            objective="l1", n_estimators=700, learning_rate=0.03, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=25, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1),
        "XGBoost": XGBRegressor(
            objective="reg:absoluteerror", n_estimators=700, learning_rate=0.03,
            max_depth=6, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbosity=0),
        "HistGBDT": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(loss="absolute_error", max_iter=450,
                                          learning_rate=0.04, random_state=seed)),
        "Ridge (linear)": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "MLP (neural)": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True,
                         n_iter_no_change=5, random_state=seed)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--sample-rows", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    pairs, X, attempted, fams = build(a.domain, a.topk, a.sample_rows)
    y = pairs["true_gain"].to_numpy(float)
    print(f"########## {a.domain} (MODEL CLASS COMPARISON) ##########", flush=True)
    print(f"records: {len(pairs):,}  organizations: {pairs.company_reference.nunique():,}  "
          f"features: {X.shape[1]}", flush=True)

    runs: dict[str, list] = {k: [] for k in models(0)}
    runs["predict zero"] = []
    for s in range(a.seeds):
        rng = np.random.default_rng(42 + s)
        comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
        trc = set(comps[:int(0.70 * len(comps))])
        tr = pairs.company_reference.isin(trc).to_numpy(); te = ~tr
        for name, mdl in models(42 + s).items():
            mdl.fit(X[tr], y[tr])
            runs[name].append(mae(mdl.predict(X)[te], y[te]))
        runs["predict zero"].append(mae(np.zeros(te.sum()), y[te]))
        print(f"  seed {s+1}/{a.seeds} done", flush=True)

    print(f"\n  organization-held-out, {a.seeds} seeds")
    print(f"  {'model':22s} {'MAE (mean +/- sd)':22s}")
    out = {}
    for name, vals in runs.items():
        m, sd = float(np.mean(vals)), float(np.std(vals))
        out[name] = {"mae": m, "sd": sd, "runs": vals}
        print(f"  {name:22s} {m:.3f} +/- {sd:.3f}")

    ref = out["LightGBM (reported)"]["mae"]
    worst_tree = max(out[k]["mae"] for k in ("XGBoost", "HistGBDT"))
    print(f"\n  spread across the three tree implementations: "
          f"{abs(worst_tree - ref):.3f} score points")
    print(f"  Ridge is {out['Ridge (linear)']['mae'] / ref:.2f}x the reported MAE; "
          f"MLP is {out['MLP (neural)']['mae'] / ref:.2f}x")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{a.domain}_model_classes.json"
    p.write_text(json.dumps(
        {"domain": a.domain, "n": int(len(pairs)),
         "n_orgs": int(pairs.company_reference.nunique()),
         "seeds": a.seeds, "results": out}, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
