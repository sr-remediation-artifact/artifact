#!/usr/bin/env python3
"""Validation of the inferred remediation-event label, and robustness of the headline.

An observed remediation event is inferred from a fall in an action family's badness
between two consecutive observations. That inference uses the later observation, so
three questions must be answered before any prospective claim rests on it:

  1. How often could the fall be explained by assets leaving the scope rather than by
     findings being fixed on assets that remain? (confound quantification)
  2. Does the headline result survive on the subset where assets did not shrink, so
     the fall cannot be attributed to disappearance? (confound removal)
  3. Does the headline result depend on the threshold used to call a fall an event?
     (specification sensitivity)

It also reports a paired, organization-clustered confidence interval for the
improvement over predicting no change, which is the quantity the claim actually needs,
and a feature ablation isolating where the signal comes from.

Usage:
  python3 audit/scripts/attempt_label_validation.py --domain web
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
from prospective_impact import build, mae  # noqa: E402

OUT = ROOT / "outputs" / "attempt_label_validation"


def fit_predict(X, y, tr, seed, cols=None):
    Xc = X if cols is None else X[cols]
    m = LGBMRegressor(objective="l1", n_estimators=700, learning_rate=0.03, num_leaves=63,
                      subsample=0.85, colsample_bytree=0.85, min_child_samples=25,
                      reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(Xc[tr], y[tr])
    return m.predict(Xc)


def org_splits(pairs, seeds):
    for s in range(seeds):
        rng = np.random.default_rng(42 + s)
        c = pairs.company_reference.unique().copy(); rng.shuffle(c)
        trc = set(c[:int(0.70 * len(c))])
        tr = pairs.company_reference.isin(trc).to_numpy()
        yield s, tr, ~tr


def paired_ci(err_model, err_zero, groups, B=2000, seed=0):
    """Organization-clustered bootstrap CI for the paired MAE reduction (zero - model)."""
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        diffs[b] = err_zero[rows].mean() - err_model[rows].mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def headline(pairs, X, y, seeds=5):
    m, z = [], []
    for s, tr, te in org_splits(pairs, seeds):
        p = fit_predict(X, y, tr, 42 + s)
        m.append(mae(p[te], y[te])); z.append(mae(np.zeros(te.sum()), y[te]))
    return float(np.mean(m)), float(np.mean(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="web")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    res = {"domain": a.domain}
    pairs, X, attempted, fams = build(a.domain, 300, 40000, 1.0)
    y = pairs["true_gain"].to_numpy(float)
    print(f"\n########## {a.domain}: remediation-event label validation ##########")
    print(f"events: {len(pairs):,}  organizations: {pairs.company_reference.nunique():,}")

    # --- 1. could the badness fall be explained by assets leaving scope? ---
    da = pairs["d_assets"].to_numpy(float)
    known = ~np.isnan(da)
    shrank = known & (da < 0)
    stable = known & (da >= 0)
    print(f"\n1. ASSET-LOSS CONFOUND (n with asset counts: {known.sum():,})")
    print(f"   assets shrank : {shrank.sum():,} ({shrank.mean()*100:.1f}%)  <- fall could be disappearance")
    print(f"   assets stable/grew: {stable.sum():,} ({stable.mean()*100:.1f}%)  <- fall must be real fixes")
    res["asset_confound"] = {"n_known": int(known.sum()), "share_shrank": float(shrank.mean()),
                             "share_stable": float(stable.mean())}

    # --- 2. does the headline survive with the confounded records removed? ---
    print(f"\n2. HEADLINE ON THE UNCONFOUNDED SUBSET (assets stable or growing)")
    pm, pz = headline(pairs, X, y, a.seeds)
    print(f"   all events          model={pm:.3f}  zero={pz:.3f}  improvement={(1-pm/pz)*100:+.0f}%")
    ps, Xs, ys = pairs[stable].reset_index(drop=True), X[stable].reset_index(drop=True), y[stable]
    sm, sz = headline(ps, Xs, ys, a.seeds)
    print(f"   assets-stable only  model={sm:.3f}  zero={sz:.3f}  improvement={(1-sm/sz)*100:+.0f}%   n={len(ps):,}")
    res["subset_check"] = {"all": {"model": pm, "zero": pz},
                           "assets_stable": {"model": sm, "zero": sz, "n": int(len(ps))}}

    # --- 3. does the result depend on the threshold that defines an event? ---
    print(f"\n3. EVENT-THRESHOLD SENSITIVITY")
    thr_res = {}
    for thr in (0.5, 1.0, 2.0):
        pt, Xt, at, _ = build(a.domain, 300, 40000, thr)
        yt = pt["true_gain"].to_numpy(float)
        tm, tz = headline(pt, Xt, yt, 3)
        thr_res[str(thr)] = {"n": int(len(pt)), "model": tm, "zero": tz}
        print(f"   drop>={thr}: n={len(pt):>7,}  model={tm:.3f}  zero={tz:.3f}  "
              f"improvement={(1-tm/tz)*100:+.0f}%")
    res["threshold_sensitivity"] = thr_res

    # --- 4. paired, organization-clustered CI for the improvement ---
    print(f"\n4. PAIRED IMPROVEMENT OVER PREDICT-ZERO (organization-clustered 95% CI)")
    rng = np.random.default_rng(42)
    c = pairs.company_reference.unique().copy(); rng.shuffle(c)
    trc = set(c[:int(0.70 * len(c))])
    tr = pairs.company_reference.isin(trc).to_numpy(); te = ~tr
    p = fit_predict(X, y, tr, 42)
    em, ez = np.abs(p[te] - y[te]), np.abs(y[te])
    lo, hi = paired_ci(em, ez, pairs.company_reference.to_numpy()[te])
    print(f"   MAE reduction = {ez.mean()-em.mean():+.3f} points  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"{'excludes 0' if lo > 0 else 'INCLUDES 0'}")
    res["paired_improvement"] = {"reduction": float(ez.mean()-em.mean()), "ci": [lo, hi],
                                 "significant": bool(lo > 0)}

    # --- 5. where does the signal come from? ---
    print(f"\n5. FEATURE ABLATION (same split, MAE)")
    groups = {"action only": [c for c in X.columns if c.startswith("action__")],
              "headroom only": [c for c in X.columns if c.startswith("headroom__")],
              "action + headroom": [c for c in X.columns if c.startswith(("action__", "headroom__"))],
              "config only": [c for c in X.columns if c.startswith("old__")] + ["old_score"],
              "full": list(X.columns)}
    abl = {}
    for name, cols in groups.items():
        if not cols:
            continue
        pp = fit_predict(X, y, tr, 42, cols)
        abl[name] = mae(pp[te], y[te])
        print(f"   {name:20s} MAE={abl[name]:.3f}  ({len(cols)} features)")
    print(f"   {'predict zero':20s} MAE={ez.mean():.3f}")
    res["ablation"] = abl

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.domain}.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / (a.domain + '.json')}")


if __name__ == "__main__":
    main()
