#!/usr/bin/env python3
"""Decision-oriented evaluation: does ranking by the estimator spend a budget better?

Error metrics say how close an estimate is; they do not say whether acting on it helps.
An operator with budget for k remediations wants the k that will actually move the score
most. We therefore rank held-out remediation opportunities by predicted gain, spend a
budget down that ranking, and measure the realized gain captured as a fraction of what
perfect foresight would have captured at the same budget.

Attribution has to be unambiguous for this to mean anything, so the ranked population is
restricted to single-family remediation events, where the realized gain belongs to one
action. Ranking is compared against the per-action historical mean, random order, and
(on Web) the rule-based counterfactual oracle, which is interesting because it may rank
well even though Section RQ2 shows it mispredicts magnitude badly.

Usage:
  python3 audit/scripts/decision_value.py --domain web
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain_reliability import ROOT  # noqa: E402
from prospective_impact import build  # noqa: E402

OUT = ROOT / "outputs" / "decision_value"
BUDGETS = (0.10, 0.25, 0.50)


def captured(order: np.ndarray, gain: np.ndarray, frac: float) -> float:
    """Realized gain collected by spending `frac` of the budget down `order`."""
    k = max(1, int(round(frac * len(gain))))
    return float(gain[order[:k]].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="web")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    pairs, X, attempted, fams = build(a.domain, 300, 40000, 1.0)
    y = pairs["true_gain"].to_numpy(float)
    single = (attempted.sum(axis=1) == 1).to_numpy()
    fam_of = np.array([attempted.columns[i] if s else ""
                       for i, s in zip(np.argmax(attempted.to_numpy(), axis=1), single)])
    print(f"\n########## {a.domain}: decision value ##########")
    print(f"single-family events available for ranking: {single.sum():,} of {len(pairs):,}")

    rows = []
    for s in range(a.seeds):
        rng = np.random.default_rng(42 + s)
        comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
        trc = set(comps[:int(0.70 * len(comps))])
        tr = pairs.company_reference.isin(trc).to_numpy()
        te = (~tr) & single                       # rank only attributable held-out events
        if te.sum() < 50:
            continue

        mdl = LGBMRegressor(objective="l1", n_estimators=700, learning_rate=0.03, num_leaves=63,
                            subsample=0.85, colsample_bytree=0.85, min_child_samples=25,
                            reg_lambda=1.0, random_state=42 + s, n_jobs=-1, verbose=-1)
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X)[te]
        g = y[te]

        gm = float(y[tr].mean())
        fmean = {f: (float(y[tr & attempted[f].to_numpy()].mean())
                     if (tr & attempted[f].to_numpy()).sum() >= 10 else gm) for f in fams}
        amean = np.array([fmean.get(f, gm) for f in fam_of[te]])

        orders = {
            "model": np.argsort(-pred),
            "action_mean": np.argsort(-amean),
            "random": rng.permutation(len(g)),
            "perfect": np.argsort(-g),
        }
        for b in BUDGETS:
            best = captured(orders["perfect"], g, b)
            for name, o in orders.items():
                if name == "perfect":
                    continue
                rows.append({"seed": s, "budget": b, "ranker": name,
                             "captured": captured(o, g, b),
                             "frac_of_best": captured(o, g, b) / best if best > 0 else np.nan})
        # realized gain of the top decile, in points (a ratio is undefined where the
        # population mean is near zero or negative, which happens on Web)
        k = max(1, int(0.10 * len(g)))
        rows.append({"seed": s, "budget": "top10_points", "ranker": "model",
                     "captured": float(g[orders['model'][:k]].mean()),
                     "frac_of_best": float(g.mean())})

    df = pd.DataFrame(rows)
    print("\n  fraction of achievable realized gain captured (mean over seeds):")
    print(f"  {'budget':>8} {'model':>18} {'action mean':>14} {'random':>10}")
    res = {}
    for b in BUDGETS:
        sub = df[df.budget == b]
        vals = {r: sub[sub.ranker == r]["frac_of_best"] for r in ["model", "action_mean", "random"]}
        res[str(b)] = {r: {"mean": float(v.mean()), "sd": float(v.std())} for r, v in vals.items()}
        print(f"  {b:>8.0%} {vals['model'].mean():>10.3f} +/-{vals['model'].std():.3f} "
              f"{vals['action_mean'].mean():>13.3f} {vals['random'].mean():>10.3f}")
    top = df[df.budget == "top10_points"]
    if len(top):
        res["top_decile_points"] = {"selected_mean": float(top["captured"].mean()),
                                    "population_mean": float(top["frac_of_best"].mean())}
        print(f"\n  mean realized gain, model's top 10%: {top['captured'].mean():+.3f} points "
              f"vs population mean {top['frac_of_best'].mean():+.3f} points")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.domain}.json").write_text(json.dumps(
        {"domain": a.domain, "n_single_family": int(single.sum()), "n_total": int(len(pairs)),
         "budgets": res}, indent=2))
    print(f"\nwrote {OUT / (a.domain + '.json')}")


if __name__ == "__main__":
    main()
