#!/usr/bin/env python3
"""Per-action prospective accuracy and the risk-coverage curve.

Two reporting gaps that the earlier scripts did not cover:

  1. Per-action prospective performance. The paper reports per-action numbers only for the
     rule-based oracle, so it is not visible whether the learned estimator's advantage is
     carried by one or two families. Here each family is evaluated on the held-out
     single-family events that belong to it, against the same predict-zero reference.

  2. The risk-coverage curve. Section RQ3 reports three operating points chosen by the
     calibration criterion; the full trade-off between how much is answered and how often
     the answer is wrong is more informative and is what an operator would tune against.

Usage:
  python3 audit/scripts/per_action_and_risk_curve.py --domain web
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain_reliability import ROOT  # noqa: E402
from prospective_impact import build, mae  # noqa: E402

OUT = ROOT / "outputs" / "per_action_risk_curve"
COVERAGES = np.arange(0.05, 1.01, 0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="web")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    pairs, X, attempted, fams = build(a.domain, 300, 40000, 1.0)
    y = pairs["true_gain"].to_numpy(float)
    single = (attempted.sum(axis=1) == 1).to_numpy()
    print(f"\n########## {a.domain}: per-action accuracy and risk-coverage ##########")

    per = {f: {"mae_model": [], "mae_zero": [], "n": []} for f in fams}
    curves = []
    for s in range(a.seeds):
        rng = np.random.default_rng(42 + s)
        comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
        n = len(comps); b1, b2 = int(.55 * n), int(.75 * n)
        g_ = pairs.company_reference.isin(set(comps[:b1])).to_numpy()
        c_ = pairs.company_reference.isin(set(comps[b1:b2])).to_numpy()
        t_ = pairs.company_reference.isin(set(comps[b2:])).to_numpy()

        mdl = LGBMRegressor(objective="l1", n_estimators=700, learning_rate=0.03, num_leaves=63,
                            subsample=0.85, colsample_bytree=0.85, min_child_samples=25,
                            reg_lambda=1.0, random_state=42 + s, n_jobs=-1, verbose=-1)
        mdl.fit(X[g_], y[g_])
        pred = mdl.predict(X)

        # --- per action, on attributable held-out events ---
        for f in fams:
            m = t_ & single & attempted[f].to_numpy()
            if m.sum() < 20:
                continue
            per[f]["mae_model"].append(mae(pred[m], y[m]))
            per[f]["mae_zero"].append(mae(np.zeros(int(m.sum())), y[m]))
            per[f]["n"].append(int(m.sum()))

        # --- risk-coverage curve ---
        err = (np.abs(pred - y) > 1.0).astype(int)
        pos = max(1, int(err[c_].sum())); neg = max(1, int((~err[c_].astype(bool)).sum()))
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.85,
                             colsample_bytree=0.85, min_child_samples=25, scale_pos_weight=neg / pos,
                             reg_lambda=1.0, random_state=42 + s, n_jobs=-1, verbose=-1)
        clf.fit(X[c_], err[c_])
        risk = clf.predict_proba(X)[:, 1][t_]
        et = err[t_]
        order = np.argsort(risk)
        curves.append([float(et[order[:max(1, int(cv * len(et)))]].mean()) for cv in COVERAGES])

    print("\n  per-action prospective accuracy (held-out single-family events)")
    print(f"  {'action':24s} {'n':>6} {'ours':>8} {'zero':>8} {'better':>8}")
    per_out = {}
    for f, v in per.items():
        if not v["n"]:
            continue
        m_, z_, n_ = np.mean(v["mae_model"]), np.mean(v["mae_zero"]), np.mean(v["n"])
        per_out[f] = {"n_mean": float(n_), "mae_model": float(m_), "mae_zero": float(z_),
                      "improvement": float(1 - m_ / z_) if z_ > 0 else None}
        print(f"  {f:24s} {n_:>6.0f} {m_:>8.3f} {z_:>8.3f} {(1-m_/z_)*100:>7.0f}%")

    cur = np.array(curves).mean(axis=0)
    print("\n  risk-coverage curve (mean realized failure rate at each coverage)")
    for cv, e in zip(COVERAGES, cur):
        if abs(cv * 100 % 10) < 1e-6:
            print(f"    coverage {cv:>4.0%}  error {e:>6.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.domain}.json").write_text(json.dumps(
        {"domain": a.domain, "per_action": per_out,
         "risk_coverage": {"coverage": [float(c) for c in COVERAGES],
                           "error": [float(e) for e in cur]}}, indent=2))
    print(f"\nwrote {OUT / (a.domain + '.json')}")


if __name__ == "__main__":
    main()
