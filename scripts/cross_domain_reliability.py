#!/usr/bin/env python3
"""Uniform cross-domain reliability replication (Web / Mail / Vulnerability).

One code path applied identically to every domain so the per-domain numbers are
directly comparable (this is the unified successor to the Web-only production
pipeline + the scratchpad Mail/Vuln reimplementation):

  1. Transition pairs = consecutive per-company snapshots ordered by date;
       ``true_gain = new_score - old_score`` (internal 0-100 score).
  2. Company-held-out split 60/20/20 (train / calibration / test), fixed seed.
  3. Direct action-gain models: LGBM, XGB, HistGBDT, residual-over-surrogate-delta.
       Baseline ``surrogate_delta`` = score-surrogate(new) - score-surrogate(old),
       surrogate trained on TRAIN snapshots only.
  4. Confidence gate: target ``|best_model_error| > 1.0 pt``, trained on CALIBRATION
       pairs (out-of-sample for the gain model), evaluated on TEST (ROC AUC).
  5. Single-action isolation: pairs where one action family's badness dropped and
       every other family stayed unchanged (``gain >= 0.5``); per-action reliability AUC.

Feature reads are bounded (dtype probe -> variance ranking on a row sample -> top-K),
so the 4.2 GB Web table is tractable via ``usecols``. Web action families are built
from the same ACTION_RULES machinery the production single-action validation uses.

Usage:
  python3 audit/scripts/cross_domain_reliability.py --domain {web|mail|vulnerability}
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Artifact note: in the research repository this constant and the Web action taxonomy are
# imported from internal modules. The taxonomy is derived from the vendor's rule
# definitions, and shipping it would disclose part of the scoring logic this paper argues
# must not be disclosed, so the artifact carries the severity weights (a trivial constant)
# inline and derives action families from the data schema instead.
SEVERITY_WEIGHTS = {"OK": 0.0, "INFO": 1.0, "LOW": 2.0, "MEDIUM": 3.0,
                    "HIGH": 4.0, "CRITICAL": 5.0, "UNKNOWN": 1.0}

def _root() -> Path:
    env = os.environ.get("SR_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for p in Path(__file__).resolve().parents:
        # make_synthetic.py marks the unpacked artifact, which a reviewer runs with no
        # git metadata present. It is checked first so that running these scripts from
        # inside the research repository still reads the artifact's own tables and
        # writes beside them, instead of reaching past them into the repository.
        if (p / "make_synthetic.py").exists():
            return p
        if (p / "CONTRIBUTIONS.md").exists() or (p / ".git").exists():
            return p
    raise RuntimeError(
        "project root not found: run from the unpacked artifact directory, "
        "or set SR_DATA_ROOT to the directory holding data/processed/")

ROOT = _root()
OUT_DIR = ROOT / "outputs" / "cross_domain_reliability"
IDCOLS = {"reference", "company_reference", "timestamp", "date",
          "internal_value", "value", "letter_value", "dataset"}
SEV_W = dict(SEVERITY_WEIGHTS)


def web_families() -> dict[str, list[tuple[str, float]]]:
    """Web action families.

    In the research repository these come from the vendor's documented remediation rules.
    The artifact derives them from the column schema instead, so that no part of the
    vendor's rule taxonomy is redistributed: every ``web_group_checker_<family>__sev_total_cnt_<sev>``
    column group becomes one family, weighted by severity.
    """
    table = ROOT / DOMAINS["web"]["table"]
    cols = pd.read_csv(table, nrows=1).columns
    pat = re.compile(r"^(web_group_checker_[^_]+(?:_[^_]+)*?)__sev_total_cnt_([A-Z]+)$")
    fams: dict[str, list[tuple[str, float]]] = {}
    for c in cols:
        m = pat.match(c)
        if m:
            fams.setdefault(m.group(1), []).append((c, float(SEV_W.get(m.group(2), 0.0))))
    return fams


DOMAINS = {
    "mail": {
        "table": "data/processed/mail_snapshot_features.csv",
        "families": lambda: {
            "spf":   [(f"mail_group_spf__sev_total_cnt_{s}", SEV_W[s])   for s in ["INFO", "LOW", "MEDIUM", "HIGH"]],
            "dmarc": [(f"mail_group_dmarc__sev_total_cnt_{s}", SEV_W[s]) for s in ["INFO", "LOW", "MEDIUM", "HIGH"]],
        },
    },
    "vulnerability": {
        "table": "data/processed/vulnerability_features.csv",
        "families": lambda: {
            "critical": [("vuln_cve_cnt_CRITICAL", SEV_W["CRITICAL"])],
            "high":     [("vuln_cve_cnt_HIGH", SEV_W["HIGH"])],
            "medium":   [("vuln_cve_cnt_MEDIUM", SEV_W["MEDIUM"])],
            "low":      [("vuln_cve_cnt_LOW", SEV_W["LOW"])],
        },
    },
    "web": {
        "table": "data/processed/web_snapshot_features.csv",
        "families": web_families,
    },
}


def regressors(seed):
    return {
        "direct_lgbm": LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.85,
                                     colsample_bytree=0.85, min_child_samples=25, reg_lambda=1.0,
                                     random_state=seed, n_jobs=-1, verbose=-1),
        "direct_xgb": XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.85,
                                   colsample_bytree=0.85, reg_lambda=1.0, objective="reg:squarederror",
                                   tree_method="hist", random_state=seed, n_jobs=-1),
        "direct_hist_gbdt": make_pipeline(SimpleImputer(strategy="median"),
                                          HistGradientBoostingRegressor(max_iter=450, learning_rate=0.04,
                                          max_leaf_nodes=31, l2_regularization=0.1, random_state=seed)),
        # non-tree model class, so the result is not tied to gradient boosting
        "direct_ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
    }


def reg_metrics(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float); e = yp - yt
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "within_1pt": float(np.mean(np.abs(e) <= 1.0)), "within_2pt": float(np.mean(np.abs(e) <= 2.0)),
            "r2": float(1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2)) if np.std(yt) > 0 else math.nan}


def badness(frame, fam_cols):
    out = np.zeros(len(frame))
    for col, w in fam_cols:
        if col in frame.columns:
            out += pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy() * float(w)
    return out


def select_features(table: Path, topk: int, sample_rows: int) -> list[str]:
    """Uniform bounded feature selection: numeric non-id cols, top-K by variance on a row sample."""
    head = pd.read_csv(table, nrows=1500)
    candidates = [c for c in head.columns if c not in IDCOLS and pd.api.types.is_numeric_dtype(head[c])]
    sample = pd.read_csv(table, usecols=candidates, nrows=sample_rows)
    var = sample.apply(pd.to_numeric, errors="coerce").var(numeric_only=True).sort_values(ascending=False)
    return [c for c in var.index[:topk]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--sample-rows", type=int, default=40000)
    ap.add_argument("--min-abs-gain", type=float, default=0.0,
                    help="Keep only transitions with |true_gain| >= this (regime filter to drop no-ops).")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = DOMAINS[a.domain]; table = ROOT / cfg["table"]; fams = cfg["families"]()
    print(f"\n########## DOMAIN = {a.domain} ##########", flush=True)

    sel = select_features(table, a.topk, a.sample_rows)
    badness_cols = sorted({c for cols in fams.values() for c, _ in cols})
    id_present = [c for c in ["reference", "company_reference", "date", "internal_value"] if c]
    usecols = list(dict.fromkeys(id_present + sel + badness_cols))
    df = pd.read_csv(table, usecols=lambda c: c in usecols)
    df = df.dropna(subset=["company_reference", "date", "internal_value", "reference"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce"); df = df.dropna(subset=["date"])
    for c in sel:
        if c not in df.columns:
            df[c] = 0.0
    print(f"snapshots={len(df):,} companies={df.company_reference.nunique():,} "
          f"features={len(sel)} families={list(fams)}", flush=True)

    for fam, cols in fams.items():
        df[f"_bad_{fam}"] = badness(df, cols)

    df = df.sort_values(["company_reference", "date"]).reset_index(drop=True)
    g = df.groupby("company_reference", sort=False)
    new = g.shift(-1)
    keep = new["reference"].notna().to_numpy()
    old = df[keep].reset_index(drop=True); new = new[keep].reset_index(drop=True)
    pairs = pd.DataFrame({"company_reference": old["company_reference"].values,
                          "old_reference": old["reference"].values, "new_reference": new["reference"].values,
                          "old_score": old["internal_value"].astype(float).values,
                          "new_score": new["internal_value"].astype(float).values})
    pairs["true_gain"] = pairs["new_score"] - pairs["old_score"]
    for fam in fams:
        pairs[f"{fam}_drop"] = old[f"_bad_{fam}"].values - new[f"_bad_{fam}"].values
    n_all = len(pairs)
    if a.min_abs_gain > 0:
        pairs = pairs[pairs["true_gain"].abs() >= a.min_abs_gain].reset_index(drop=True)
    print(f"transition pairs={len(pairs):,} (of {n_all:,}, |gain|>={a.min_abs_gain}) "
          f"improving={(pairs.true_gain>0).sum():,} gain mean={pairs.true_gain.mean():.2f} "
          f"p90={pairs.true_gain.quantile(.9):.2f}", flush=True)

    fidx = df.set_index("reference")
    Xo = fidx.reindex(pairs["old_reference"].values)[sel].reset_index(drop=True)
    Xn = fidx.reindex(pairs["new_reference"].values)[sel].reset_index(drop=True)
    Xo_np, Xn_np = Xo.to_numpy(dtype=float), Xn.to_numpy(dtype=float)
    X = pd.DataFrame(np.hstack([Xo_np, Xn_np, Xn_np - Xo_np]),
                     columns=[f"old__{c}" for c in sel] + [f"new__{c}" for c in sel] + [f"delta__{c}" for c in sel])
    X["old_score"] = pairs["old_score"].values
    print(f"feature matrix={X.shape[0]:,} x {X.shape[1]:,}", flush=True)

    # Four-way company-held-out split so the guarantee is honest:
    #   g = train the gain model | c = train the confidence model (gain-model out-of-sample)
    #   k = set the conformal threshold (confidence-model out-of-sample) | t = final test (unseen by all)
    comps = pairs["company_reference"].unique().copy()
    np.random.default_rng(a.seed).shuffle(comps)
    n = len(comps)
    b1, b2, b3 = int(.55 * n), int(.75 * n), int(.85 * n)
    g_, c_, k_, t_ = set(comps[:b1]), set(comps[b1:b2]), set(comps[b2:b3]), set(comps[b3:])
    split = pairs["company_reference"].map(
        lambda co: "g" if co in g_ else "c" if co in c_ else "k" if co in k_ else "t").to_numpy()
    m = {s: (split == s) for s in ["g", "c", "k", "t"]}
    oos = m["k"] | m["t"]  # out-of-sample for BOTH gain and confidence models
    print(f"split rows gain={m['g'].sum():,} conf={m['c'].sum():,} thresh={m['k'].sum():,} test={m['t'].sum():,}", flush=True)

    y = pairs["true_gain"].to_numpy(float)
    # surrogate-delta baseline: score surrogate trained on gain-train snapshots
    surro = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                          min_child_samples=25, reg_lambda=1.0, random_state=a.seed, n_jobs=-1, verbose=-1)
    train_refs = set(pairs.loc[m["g"], "old_reference"]).union(pairs.loc[m["g"], "new_reference"])
    strain = df[df["reference"].isin(train_refs)]
    surro.fit(strain[sel].astype(float), strain["internal_value"].astype(float))
    base = surro.predict(pd.DataFrame(Xn_np, columns=sel)) - surro.predict(pd.DataFrame(Xo_np, columns=sel))
    preds = {"surrogate_delta": np.asarray(base, float)}
    for name, mdl in regressors(a.seed).items():
        mdl.fit(X.loc[m["g"]], y[m["g"]]); preds[name] = np.asarray(mdl.predict(X), float)
    res = LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                        min_child_samples=25, reg_lambda=1.0, random_state=a.seed, n_jobs=-1, verbose=-1)
    res.fit(X.loc[m["g"]], (y - preds["surrogate_delta"])[m["g"]])
    preds["residual_lgbm"] = preds["surrogate_delta"] + res.predict(X)
    # floor baselines (naive references so the comparison is not only vs our own surrogate)
    preds["predict_zero"] = np.zeros(len(pairs))
    preds["predict_mean"] = np.full(len(pairs), float(y[m["g"]].mean()))
    # per-action historical mean: the practitioner heuristic "this action is usually worth X".
    # Tests whether the model does more than memorize the identity of the action family.
    train_mean = float(y[m["g"]].mean())
    drops = {fam: pairs[f"{fam}_drop"].to_numpy(float) >= 1.0 for fam in fams}
    fam_mean = {}
    for fam, d in drops.items():
        sel_rows = d & m["g"]
        fam_mean[fam] = float(y[sel_rows].mean()) if sel_rows.sum() >= 10 else train_mean
    acted = np.zeros(len(pairs)); n_act = np.zeros(len(pairs))
    for fam, d in drops.items():
        acted += d * fam_mean[fam]; n_act += d
    preds["predict_action_mean"] = np.where(n_act > 0, acted / np.maximum(n_act, 1), train_mean)
    BASELINES = {"surrogate_delta", "predict_zero", "predict_mean", "predict_action_mean"}

    print("\n=== direct-gain metrics (TEST, company-held-out) ===", flush=True)
    tm = {}
    for name, p in preds.items():
        mm = reg_metrics(y[m["t"]], p[m["t"]]); tm[name] = mm
        tag = "  [baseline]" if name in BASELINES else ""
        print(f"  {name:16s} MAE={mm['mae']:.3f} within1pt={mm['within_1pt']:.3f} R2={mm['r2']:.3f}{tag}", flush=True)
    sel_mae = {k: reg_metrics(y[m['c']], p[m['c']])['mae'] for k, p in preds.items() if k not in BASELINES}
    best = min(sel_mae, key=sel_mae.get)
    print(f"  -> best point model (selected on confidence-split MAE): {best}", flush=True)

    bad = (np.abs(preds[best] - y) > 1.0).astype(int)
    pos = max(1, int(bad[m['c']].sum())); neg = max(1, int((~bad[m['c']].astype(bool)).sum()))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.85, colsample_bytree=0.85,
                         min_child_samples=25, reg_lambda=1.0, scale_pos_weight=neg / pos,
                         random_state=a.seed, n_jobs=-1, verbose=-1)
    clf.fit(X.loc[m['c']], bad[m['c']]); risk = clf.predict_proba(X)[:, 1]

    def auc_of(mask):
        yy = bad[mask]; n_ = int(mask.sum()); p_ = int(yy.sum())
        if p_ == 0 or p_ == n_:
            return {"n": n_, "pos": p_, "auc": None}
        return {"n": n_, "pos": p_, "pos_rate": p_ / n_, "auc": float(roc_auc_score(yy, risk[mask])),
                "ap": float(average_precision_score(yy, risk[mask])),
                "brier": float(brier_score_loss(yy, np.clip(risk[mask], 0, 1)))}

    full = auc_of(m['t'])

    # sensitivity of the reliability signal to the error tolerance tau
    tau_sweep = {}
    for tau in (0.5, 1.0, 2.0):
        bad_t = (np.abs(preds[best] - y) > tau).astype(int)
        p_c = max(1, int(bad_t[m['c']].sum())); n_c = max(1, int((~bad_t[m['c']].astype(bool)).sum()))
        clf_t = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.85,
                               colsample_bytree=0.85, min_child_samples=25, scale_pos_weight=n_c / p_c,
                               reg_lambda=1.0, random_state=a.seed, n_jobs=-1, verbose=-1)
        clf_t.fit(X.loc[m['c']], bad_t[m['c']])
        r_t = clf_t.predict_proba(X)[:, 1]
        yy = bad_t[m['t']]
        tau_sweep[str(tau)] = ({"auc": float(roc_auc_score(yy, r_t[m['t']])),
                                "pos_rate": float(yy.mean())} if 0 < yy.sum() < len(yy) else None)
    print("  tau sensitivity (test AUC):",
          {k: (round(v['auc'], 3) if v else None) for k, v in tau_sweep.items()}, flush=True)

    # bootstrap 95% CI on the full-split test AUC
    def boot_auc_ci(mask, B=1000):
        yy = bad[mask]; rr = risk[mask]
        if len(np.unique(yy)) < 2:
            return [None, None]
        rng = np.random.default_rng(0); idx = np.arange(len(yy)); vals = []
        for _ in range(B):
            s = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(yy[s])) < 2:
                continue
            vals.append(roc_auc_score(yy[s], rr[s]))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    # Conformal-style selective-acceptance guarantee (finite-sample Wilson upper bound).
    # Pick, on the THRESHOLD split k (unseen by the confidence model), the largest risk
    # threshold whose accepted-set failure rate has Wilson upper bound <= target; apply to
    # the untouched TEST split t and report coverage + empirical failure rate.
    def wilson_upper(k, n, z=1.96):
        if n == 0:
            return 1.0
        p = k / n
        return (p + z * z / (2 * n) + z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)

    def risk_control(targets):
        cal_bad = bad[m['k']]; cal_risk = risk[m['k']]
        order = np.argsort(cal_risk, kind="mergesort")
        sb = cal_bad[order]; sr = cal_risk[order]; cum = np.cumsum(sb)
        test_bad = bad[m['t']]; test_risk = risk[m['t']]
        rows = []
        for t in targets:
            thr = None
            for i in range(len(sb)):
                if wilson_upper(int(cum[i]), i + 1) <= t:
                    thr = float(sr[i])
            if thr is None:
                rows.append({"target_risk": t, "threshold": None, "test_coverage": 0.0,
                             "test_empirical_risk": None, "guarantee_held": None, "n_accepted": 0})
                continue
            acc = test_risk <= thr
            emp = float(test_bad[acc].mean()) if acc.any() else None
            rows.append({"target_risk": t, "threshold": thr, "test_coverage": float(acc.mean()),
                         "test_empirical_risk": emp, "guarantee_held": (emp is not None and emp <= t),
                         "n_accepted": int(acc.sum())})
        return rows

    auc_ci = boot_auc_ci(m['t'])
    rc = risk_control([0.05, 0.10, 0.20])
    print(f"\n=== reliability gate (target |err|>1pt, best={best}) ===")
    print(f"  FULL test split  n={full['n']} pos={full['pos']} AUC={full.get('auc')} "
          f"95%CI=[{auc_ci[0]},{auc_ci[1]}]", flush=True)
    print("  selective-acceptance guarantee (accept if risk<=thr; want empirical<=target):", flush=True)
    for r in rc:
        print(f"    target<={r['target_risk']:.2f}  test_coverage={r['test_coverage']:.3f} "
              f"accepted={r['n_accepted']} empirical_risk={r['test_empirical_risk']} held={r['guarantee_held']}", flush=True)

    print(f"\n=== single-action ISOLATED (gain>=0.5; AUC on out-of-sample k+t only) ===", flush=True)
    fam_ids = list(fams); iso_all = np.zeros(len(pairs), bool); per_action = {}
    for fam in fam_ids:
        mask = (pairs[f"{fam}_drop"] >= 1.0).to_numpy() & (pairs["true_gain"] >= 0.5).to_numpy()
        for other in fam_ids:
            if other != fam:
                mask &= (pairs[f"{other}_drop"].abs() <= 0.0).to_numpy()
        iso_all |= mask
        n_total = int(mask.sum())
        emask = mask & oos
        r = auc_of(emask)
        gm = reg_metrics(y[emask], preds[best][emask]) if emask.sum() > 0 else {"mae": None, "within_1pt": None}
        r.update({"clean_total_all_splits": n_total,
                  "companies": int(pairs.loc[mask, "company_reference"].nunique()) if n_total else 0,
                  "model_mae": gm["mae"], "model_within_1pt": gm["within_1pt"]})
        per_action[fam] = r
        print(f"  {fam:14s} clean_total={n_total:4d} eval_oos={r['n']:4d} MAE={gm['mae']} "
              f"unstable={r['pos']} AUC={r.get('auc')}", flush=True)
    iso = auc_of(iso_all & oos); iso["clean_total_all_splits"] = int(iso_all.sum())
    print(f"  ALL isolated     clean_total={int(iso_all.sum())} eval_oos={iso['n']} AUC={iso.get('auc')}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"domain": a.domain, "n_pairs": int(len(pairs)), "n_companies": int(pairs.company_reference.nunique()),
              "best_point_model": best, "test_metrics": {k: tm[k] for k in tm},
              "baseline_test_mae": {k: tm[k]["mae"] for k in
                                    ("surrogate_delta", "predict_zero", "predict_mean", "predict_action_mean")},
              "full_split_reliability": full, "full_split_auc_ci95": auc_ci, "risk_control": rc,
              "tau_sensitivity": tau_sweep,
              "isolated_overall": iso, "per_action": per_action, "seed": a.seed, "topk": a.topk,
              "min_abs_gain": a.min_abs_gain, "gain_mean": float(pairs.true_gain.mean())}
    slug = "" if a.min_abs_gain == 0 else f"_ming{str(a.min_abs_gain).replace('.', '_')}"
    (OUT_DIR / f"{a.domain}{slug}_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_DIR / (a.domain + slug + '_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
