#!/usr/bin/env python3
"""Fair (prospective) comparison between a learned estimator and the rule-based oracle.

The retrospective estimator in cross_domain_reliability.py is given features of BOTH
snapshots, so it observes the realized post-remediation configuration. The rule-based
oracle is given only the pre-remediation configuration plus the intended action, which
it assumes is applied completely. Comparing the two directly is not like-for-like.

This script puts both estimators in the operator's actual decision-time information
set: the current configuration, plus which action family is about to be attempted.
Nothing about the realized outcome is available to either. Evaluation is on held-out
organizations.

Usage:
  python3 audit/scripts/prospective_vs_rules_oracle.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PRED = (ROOT / "outputs/surrogates_web_feature_ablation/runs/param_groups"
        / "rules_counterfactual_validation_all/rules_counterfactual_predictions.csv")
FEATS = ROOT / "data/processed/web_snapshot_features.csv"
OUT = ROOT / "outputs/prospective_vs_rules_oracle"
TOPK = 200
SEED = 42


def main() -> None:
    d = pd.read_csv(PRED)
    print(f"single-action transitions: {len(d):,}  organizations: {d.company_reference.nunique()}")

    # old-snapshot features only, selected by variance on the same table
    head = pd.read_csv(FEATS, nrows=5000)
    drop = {"reference", "company_reference", "date", "internal_value"}
    num = [c for c in head.columns if c not in drop and pd.api.types.is_numeric_dtype(head[c])]
    var = head[num].var(numeric_only=True).sort_values(ascending=False)
    sel = list(var.head(TOPK).index)
    feats = pd.read_csv(FEATS, usecols=lambda c: c in set(sel) | {"reference"})
    d = d.merge(feats.add_prefix("old__").rename(columns={"old__reference": "old_reference"}),
                on="old_reference", how="left")
    old_cols = [f"old__{c}" for c in sel if f"old__{c}" in d.columns]

    # the intended action is chosen by the operator, so its identity is legitimate input;
    # its realized extent is not, and is never supplied.
    act = pd.get_dummies(d["action_id"], prefix="act")
    # per-family remediation headroom implied by the CURRENT configuration: the single
    # most informative prospective signal, since gain scales with what there is to fix.
    head_cols = [c for c in d.columns if c.endswith("__old_badness")]
    X = pd.concat([d[old_cols].astype(float), d[head_cols].astype(float),
                   d[["old_true_score"]].astype(float), act], axis=1)
    y = d["true_gain"].to_numpy(float)

    rng = np.random.default_rng(SEED)
    comps = d.company_reference.unique().copy()
    rng.shuffle(comps)
    cut = int(0.65 * len(comps))
    train_c = set(comps[:cut])
    tr = d.company_reference.isin(train_c).to_numpy()
    te = ~tr
    print(f"held-out split: train {tr.sum():,} rows / {len(train_c)} orgs   "
          f"test {te.sum():,} rows / {len(comps) - cut} orgs")

    mdl = LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.85,
                        colsample_bytree=0.85, min_child_samples=20, reg_lambda=1.0,
                        random_state=SEED, n_jobs=-1, verbose=-1)
    mdl.fit(X[tr], y[tr])
    pro = mdl.predict(X)

    # per-action historical mean, fit on the same training organizations
    gm = float(y[tr].mean())
    amean = d["action_id"].map(pd.Series(y[tr]).groupby(d.loc[tr, "action_id"].values).mean()).fillna(gm).to_numpy()

    est = {"prospective_model": pro, "rules_oracle": d["rules_gain"].to_numpy(float),
           "action_mean": amean, "predict_zero": np.zeros(len(d))}

    def mae(p, m):
        return float(np.mean(np.abs(p[m] - y[m])))

    def w1(p, m):
        return float(np.mean(np.abs(p[m] - y[m]) <= 1.0))

    def bias(p, m):
        return float(np.mean(p[m] - y[m]))

    print("\n=== HELD-OUT ORGANIZATIONS, decision-time information only ===")
    res = {}
    for k, p in est.items():
        res[k] = {"mae": mae(p, te), "within_1pt": w1(p, te), "bias": bias(p, te)}
        print(f"  {k:20s} MAE={res[k]['mae']:.3f}  within1pt={res[k]['within_1pt']*100:5.1f}%  "
              f"bias={res[k]['bias']:+.3f}")

    print("\n=== per action (held-out) ===")
    per = {}
    for a, gsub in d[te].groupby("action_id"):
        m = np.zeros(len(d), bool)
        m[gsub.index] = True
        if m.sum() < 15:
            continue
        per[a] = {"n": int(m.sum()), "true_mean": float(y[m].mean()),
                  "rules_mean": float(est["rules_oracle"][m].mean()),
                  "rules_mae": mae(est["rules_oracle"], m),
                  "model_mae": mae(est["prospective_model"], m)}
        print(f"  {a:22s} n={per[a]['n']:4d}  true={per[a]['true_mean']:+5.2f}  "
              f"rules={per[a]['rules_mean']:+5.2f}  rulesMAE={per[a]['rules_mae']:5.2f}  "
              f"modelMAE={per[a]['model_mae']:5.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prospective_vs_rules.json").write_text(json.dumps(
        {"n_rows": int(len(d)), "n_test_rows": int(te.sum()),
         "n_train_orgs": len(train_c), "n_test_orgs": int(len(comps) - cut),
         "overall": res, "per_action": per, "topk": TOPK, "seed": SEED}, indent=2))
    print(f"\nwrote {OUT / 'prospective_vs_rules.json'}")


if __name__ == "__main__":
    main()
