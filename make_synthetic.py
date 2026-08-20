#!/usr/bin/env python3
"""Generate a synthetic dataset with the same schema as the proprietary tables.

The underlying observations describe real, identifiable organizations and cannot be
released. This generator produces a stand-in with the same column names, dtypes, and
longitudinal structure, so a reviewer can execute the full pipeline end to end and
confirm that it runs, that the splits are organization-disjoint, and that every reported
statistic is computed the way the paper describes.

The synthetic data is NOT drawn from the real distribution and will not reproduce the
paper's numbers. It exists to make the code executable and auditable, not to stand in
for the measurement.

Usage:
  python3 make_synthetic.py --out data/processed
  python3 scripts/prospective_impact.py --domain web     # then runs against it
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = {
    "web": {"file": "web_snapshot_features.csv", "orgs": 300, "families": [
        "http-to-https", "hsts", "referrer-policy", "csp-baseline", "xfo-xcto",
        "certificate-rotation", "tls-legacy", "weak-ciphers", "ssl-vulns"]},
    "mail": {"file": "mail_snapshot_features.csv", "orgs": 200, "families": ["spf", "dmarc"]},
    "vulnerability": {"file": "vulnerability_features.csv", "orgs": 250,
                      "families": ["critical", "high", "medium", "low"]},
}
SEV = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def badness_cols(domain: str, fam: str) -> list[str]:
    if domain == "web":
        return [f"web_group_checker_{fam.replace('-', '')}__sev_total_cnt_{s}" for s in SEV]
    if domain == "mail":
        return [f"mail_group_{fam}__sev_total_cnt_{s}" for s in ["INFO", "LOW", "MEDIUM", "HIGH"]]
    return [f"vuln_cve_cnt_{fam.upper()}"]


def build(domain: str, rng: np.random.Generator) -> pd.DataFrame:
    spec = SPEC[domain]
    rows = []
    for o in range(spec["orgs"]):
        org = f"synth-org-{o:04d}"
        n_obs = int(rng.integers(4, 14))
        state = {c: float(rng.integers(0, 40)) for f in spec["families"] for c in badness_cols(domain, f)}
        n_assets = float(rng.integers(5, 300))
        score = float(rng.uniform(40, 95))
        for k in range(n_obs):
            # some periods see findings resolved, which is what the pipeline detects
            for f in spec["families"]:
                if rng.random() < 0.25:
                    for c in badness_cols(domain, f):
                        state[c] = max(0.0, state[c] - float(rng.integers(0, 8)))
            n_assets = max(1.0, n_assets + float(rng.integers(-3, 6)))
            score = float(np.clip(score + rng.normal(0.3, 1.5), 0, 100))
            rows.append({"reference": f"{org}-snap-{k:03d}", "company_reference": org,
                         "date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * k)).date(),
                         "internal_value": round(score, 3), "n_assets": n_assets, **state})
    df = pd.DataFrame(rows)
    # a few generic numeric columns so the variance-based feature selector has choices
    for i in range(40):
        df[f"synth_feature_{i:02d}"] = rng.normal(size=len(df)) * (i + 1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for domain, spec in SPEC.items():
        df = build(domain, rng)
        path = out / spec["file"]
        df.to_csv(path, index=False)
        print(f"{domain:14s} -> {path}  ({len(df):,} rows, {df.company_reference.nunique()} organizations)")
    print("\nSynthetic only. It exercises the pipeline; it does not reproduce the paper's numbers.")


if __name__ == "__main__":
    main()
