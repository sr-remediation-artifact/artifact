#!/usr/bin/env python3
"""Export the measured quantities the demo displays, so nothing in it is invented.

Everything the page shows is either read from an existing result JSON or computed here
from the same pipeline the paper uses. The one new quantity is the response of realized
gain to remediation headroom, binned per action family, which is the mechanism the
estimator relies on and the thing the demo lets a reader vary.

Usage:
  python3 audit/scripts/export_demo_data.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain_reliability import ROOT  # noqa: E402
from prospective_impact import build  # noqa: E402

OUT = ROOT / "demo"
NBINS = 6


def load(p):
    f = ROOT / p
    return json.load(open(f)) if f.exists() else None


def main() -> None:
    demo = {"domains": {}, "sources": {}}

    for d in ["web", "mail", "vulnerability"]:
        pairs, X, attempted, fams = build(d, 300, 40000, 1.0)
        y = pairs["true_gain"].to_numpy(float)
        single = (attempted.sum(axis=1) == 1).to_numpy()

        fam_curves = {}
        for f in fams:
            m = single & attempted[f].to_numpy()
            if m.sum() < 60:
                continue
            h = X.loc[m, f"headroom__{f}"].to_numpy(float)
            g = y[m]
            # quantile bins so each point rests on a comparable number of records
            qs = np.unique(np.quantile(h, np.linspace(0, 1, NBINS + 1)))
            if len(qs) < 3:
                continue
            pts = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                sel = (h >= lo) & (h <= hi if hi == qs[-1] else h < hi)
                if sel.sum() < 15:
                    continue
                pts.append({"headroom": float(np.median(h[sel])),
                            "gain": float(np.mean(g[sel])),
                            "n": int(sel.sum())})
            if len(pts) >= 3:
                fam_curves[f] = {"points": pts,
                                 "headroom_max": float(np.quantile(h, 0.98)),
                                 "n": int(m.sum()),
                                 "realized_mean": float(g.mean())}

        acc = load(f"outputs/prospective_impact/{d}_orgheldout.json")
        tmp = load(f"outputs/prospective_impact/{d}_temporal.json")
        tmo = load(f"outputs/prospective_impact/{d}_temporal_orgs.json")
        dec = load(f"outputs/decision_value/{d}.json")
        pac = load(f"outputs/per_action_risk_curve/{d}.json")

        demo["domains"][d] = {
            "n_records": int(len(pairs)),
            "n_orgs": int(pairs.company_reference.nunique()),
            "n_families": len(fams),
            "mean_gain": float(y.mean()),
            "share_small": float(np.mean(np.abs(y) < 0.5)),
            "single_family_share": float(single.mean()),
            "headroom_response": fam_curves,
            "accuracy": acc["results"] if acc else None,
            "paired": acc["paired_improvement"] if acc else None,
            "reliability_auc": acc.get("reliability_auc_mean") if acc else None,
            "calibration": acc.get("risk_calibration") if acc else None,
            "temporal": {"date_only": tmp["results"] if tmp else None,
                         "date_and_org": tmo["results"] if tmo else None},
            "decision": dec["budgets"] if dec else None,
            "per_action": pac["per_action"] if pac else None,
            "risk_curve": pac["risk_coverage"] if pac else None,
        }
        print(f"{d:14s} records={len(pairs):,} families_with_curves={len(fam_curves)}")

    orc = load("outputs/prospective_vs_rules_oracle/prospective_vs_rules.json")
    demo["oracle"] = {"overall": orc["overall"], "per_action": orc["per_action"],
                      "n_test": orc["n_test_rows"], "n_orgs": orc["n_test_orgs"]} if orc else None
    val = load("outputs/attempt_label_validation/web.json")
    demo["validity"] = val

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "demo_data.json").write_text(json.dumps(demo, indent=1))
    kb = (OUT / "demo_data.json").stat().st_size / 1024
    print(f"\nwrote {OUT/'demo_data.json'} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
