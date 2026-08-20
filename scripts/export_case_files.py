#!/usr/bin/env python3
"""Export per-organization case files: what the operator is shown, and what actually happened.

The demo needs to walk a reader through one concrete decision, so this exports held-out
organizations with everything needed to reconstruct it: the configuration they were in,
the remediation opportunities open to them, what the tool would have told them before the
work, and what the score actually did afterwards.

Nothing here is simulated. Organizations, scores, headroom, and realized outcomes are the
real records; the prediction and the reliability decision come from models fitted on a
disjoint set of organizations. Organization identifiers are replaced with sequential
labels so nothing traceable is published.

Usage:
  python3 audit/scripts/export_case_files.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain_reliability import ROOT  # noqa: E402

PRED = (ROOT / "outputs/surrogates_web_feature_ablation/runs/param_groups"
        / "rules_counterfactual_validation_all/rules_counterfactual_predictions.csv")
FEATS = ROOT / "data/processed/web_snapshot_features.csv"
OUT = ROOT / "demo"
SEED, TOPK, TAU = 42, 200, 1.0
PRETTY = {"http-to-https": "Force HTTPS on plaintext assets", "hsts": "Add HSTS",
          "referrer-policy": "Set a strict referrer policy", "csp-baseline": "Ship a CSP baseline",
          "xfo-xcto": "Normalize security headers", "certificate-rotation": "Rotate weak certificates",
          "tls-legacy": "Drop legacy TLS", "weak-ciphers": "Remove weak cipher suites",
          "ssl-vulns": "Remediate SSL vulnerabilities"}


def main() -> None:
    d = pd.read_csv(PRED)
    head = pd.read_csv(FEATS, nrows=5000)
    drop = {"reference", "company_reference", "date", "internal_value"}
    num = [c for c in head.columns if c not in drop and pd.api.types.is_numeric_dtype(head[c])]
    sel = list(head[num].var(numeric_only=True).sort_values(ascending=False).head(TOPK).index)
    feats = pd.read_csv(FEATS, usecols=lambda c: c in set(sel) | {"reference"})
    d = d.merge(feats.add_prefix("old__").rename(columns={"old__reference": "old_reference"}),
                on="old_reference", how="left")

    fam_cols = [c for c in d.columns if c.endswith("__old_badness")]
    fams = [c.replace("__old_badness", "") for c in fam_cols]
    act = pd.get_dummies(d["action_id"], prefix="act")
    X = pd.concat([d[[f"old__{c}" for c in sel if f"old__{c}" in d.columns]].astype(float),
                   d[fam_cols].astype(float), d[["old_true_score"]].astype(float), act], axis=1)
    y = d["true_gain"].to_numpy(float)

    rng = np.random.default_rng(SEED)
    comps = d.company_reference.unique().copy(); rng.shuffle(comps)
    n = len(comps)
    tr_c, cal_c = set(comps[:int(.55 * n)]), set(comps[int(.55 * n):int(.75 * n)])
    tr = d.company_reference.isin(tr_c).to_numpy()
    cal = d.company_reference.isin(cal_c).to_numpy()
    te = ~(tr | cal)

    mdl = LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=31,
                        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
                        reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    mdl.fit(X[tr], y[tr]); pred = mdl.predict(X)

    err = (np.abs(pred - y) > TAU).astype(int)
    pos, neg = max(1, int(err[cal].sum())), max(1, int((~err[cal].astype(bool)).sum()))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.85,
                         colsample_bytree=0.85, min_child_samples=25, scale_pos_weight=neg / pos,
                         reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(X[cal], err[cal]); risk = clf.predict_proba(X)[:, 1]
    # accept threshold at a 10% target, set on the calibration organizations
    TARGET = 0.05
    cand = np.quantile(risk[cal], np.linspace(0.05, 1.0, 60))
    thr = float(cand[0])
    for c_ in cand:
        a = risk[cal] <= c_
        if a.sum() >= 20 and err[cal][a].mean() <= TARGET:
            thr = float(c_)
    print(f"calibration: base error {err[cal].mean():.1%}, threshold {thr:.3f} "
          f"at a {TARGET:.0%} target, accepting {(risk[cal] <= thr).mean():.0%} of calibration cases")

    # per-action historical means, from training organizations only
    fam_mean = {}
    for f in fams:
        m = tr & (d["action_id"] == f).to_numpy()
        fam_mean[f] = float(y[m].mean()) if m.sum() >= 10 else float(y[tr].mean())

    cases = {}
    dte = d[te].copy()
    dte["_pred"], dte["_risk"], dte["_err"] = pred[te], risk[te], err[te]
    for i, (comp, g) in enumerate(dte.groupby("company_reference")):
        if len(g) < 2:
            continue
        label = f"ORG-{i+1:02d}"
        row0 = g.iloc[0]
        opps = []
        g = (g.sort_values("_risk").groupby("action_id", as_index=False).first())
        if len(g) < 2:
            continue
        for _, r in g.iterrows():
            opps.append({
                "family": r["action_id"],
                "label": PRETTY.get(r["action_id"], r["action_id"]),
                "headroom": float(r.get(f"{r['action_id']}__old_badness", 0.0)),
                "predicted": float(r["_pred"]),
                "risk": float(r["_risk"]),
                "accepted": bool(r["_risk"] <= thr),
                "rules_claim": float(r["rules_gain"]),
                "realized": float(r["true_gain"]),
                "action_mean": fam_mean.get(r["action_id"]),
                "risk_pct": float(round(r["_risk"] * 100)),
                "score_before": float(r["old_true_score"]),
                "score_after": float(r["new_true_score"]),
            })
        opps.sort(key=lambda o: -o["predicted"])
        cases[label] = {
            "label": label,
            "score": float(row0["old_true_score"]),
            "n_opportunities": len(opps),
            "headroom": {f: float(row0.get(f"{f}__old_badness", 0.0)) for f in fams
                         if float(row0.get(f"{f}__old_badness", 0.0)) > 0},
            "opportunities": opps,
        }
        if len(cases) >= 200:
            break

    # Two groups, and the demo labels which is which. The illustrative ones are ranked by how
    # much better our estimate is than the rules on the same options, so they show the contrast
    # the paper is about. The random ones are an unfiltered draw from everything else, so a
    # reader can see typical behaviour next to the selected behaviour.
    def contrast(c):
        o = c["opportunities"]
        ours = sum(abs(x["predicted"] - x["realized"]) for x in o) / len(o)
        rules = sum(abs(x["rules_claim"] - x["realized"]) for x in o) / len(o)
        return rules - ours
    pool = [c for c in cases.values() if c["n_opportunities"] >= 2]
    # the walkthrough reads better with a real choice, so illustrative cases need three options
    picked = sorted([c for c in pool if c["n_opportunities"] >= 3], key=lambda c: -contrast(c))[:10]
    for c in picked:
        c["selection"] = "illustrative"
    rest = [c for c in pool if c not in picked]
    rng2 = np.random.default_rng(7)
    rnd = [rest[i] for i in rng2.choice(len(rest), size=min(5, len(rest)), replace=False)] if rest else []
    for c in rnd:
        c["selection"] = "random"
    ordered = picked + rnd
    out = {"threshold": thr, "tau": TAU, "target": 0.05,
           "n_illustrative": sum(1 for c in ordered if c["selection"]=="illustrative"),
           "n_random": sum(1 for c in ordered if c["selection"]=="random"),
           "n_pool": len(pool),
           "n_train_orgs": len(tr_c), "n_cal_orgs": len(cal_c),
           "n_test_orgs": int(dte.company_reference.nunique()),
           "pretty": PRETTY, "cases": {c["label"]: c for c in ordered}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "case_files.json").write_text(json.dumps(out, indent=1))
    tot = sum(c["n_opportunities"] for c in ordered)
    print(f"wrote {OUT/'case_files.json'}: {len(ordered)} organizations, {tot} decisions")


if __name__ == "__main__":
    main()
