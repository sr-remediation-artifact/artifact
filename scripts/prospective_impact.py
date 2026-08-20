#!/usr/bin/env python3
"""Fully prospective remediation-impact estimation.

Every feature here is available to an operator BEFORE remediation is executed:
the current configuration, the per-family remediation headroom implied by it, and
the identity of the action family about to be attempted. Nothing derived from the
post-remediation configuration is supplied. This is the setting the operational
question actually poses, and it is the setting in which the estimator must beat the
zero-effect and per-action-mean references to be worth deploying.

Population: transitions in which at least one action family was attempted, i.e. its
badness fell. No filter on the realized gain is applied, so the evaluation is not
conditioned on the answer.

Usage:
  python3 audit/scripts/prospective_impact.py --domain web
  python3 audit/scripts/prospective_impact.py --domain web --temporal
"""
from __future__ import annotations
import argparse, json, warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_domain_reliability import DOMAINS, ROOT, badness, select_features  # noqa: E402

OUT = ROOT / "outputs" / "prospective_impact"


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p, float) - y)))


def build(domain: str, topk: int, sample_rows: int, drop_thr: float = 1.0):
    spec = DOMAINS[domain]
    table = ROOT / spec["table"]
    fams = spec["families"]()
    sel = select_features(table, topk, sample_rows)
    bad_cols = sorted({c for cols in fams.values() for c, _ in cols})
    use = list(dict.fromkeys(["reference", "company_reference", "date", "internal_value", "n_assets"] + sel + bad_cols))
    df = pd.read_csv(table, usecols=lambda c: c in use)
    df = df.dropna(subset=["company_reference", "date", "internal_value", "reference"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in sel:
        if c not in df.columns:
            df[c] = 0.0
    for fam, cols in fams.items():
        df[f"_bad_{fam}"] = badness(df, cols)

    df = df.sort_values(["company_reference", "date"]).reset_index(drop=True)
    nxt = df.groupby("company_reference", sort=False).shift(-1)
    keep = nxt["reference"].notna().to_numpy()
    old, new = df[keep].reset_index(drop=True), nxt[keep].reset_index(drop=True)

    pairs = pd.DataFrame({
        "company_reference": old["company_reference"].values,
        "old_date": old["date"].values,
        "old_score": old["internal_value"].astype(float).values,
        "true_gain": new["internal_value"].astype(float).values - old["internal_value"].astype(float).values,
    })
    # PROSPECTIVE features only: current config, current per-family headroom, attempted action
    Xo = old[sel].astype(float).reset_index(drop=True)
    Xo.columns = [f"old__{c}" for c in sel]
    head = pd.DataFrame({f"headroom__{f}": old[f"_bad_{f}"].values for f in fams})
    attempted = pd.DataFrame({f: (old[f"_bad_{f}"].values - new[f"_bad_{f}"].values) >= drop_thr for f in fams})
    acted = attempted.any(axis=1).to_numpy()

    X = pd.concat([Xo, head, attempted.add_prefix("action__").astype(float),
                   pairs[["old_score"]]], axis=1)
    if "n_assets" in old.columns and "n_assets" in new.columns:
        pairs["d_assets"] = (pd.to_numeric(new["n_assets"], errors="coerce").values
                             - pd.to_numeric(old["n_assets"], errors="coerce").values)
    else:
        pairs["d_assets"] = np.nan
    return pairs[acted].reset_index(drop=True), X[acted].reset_index(drop=True), \
        attempted[acted].reset_index(drop=True), list(fams)


def evaluate(pairs, X, attempted, fams, tr, te, seed, objective='l2'):
    y = pairs["true_gain"].to_numpy(float)
    mdl = LGBMRegressor(objective=objective, n_estimators=700, learning_rate=0.03, num_leaves=63,
                        subsample=0.85, colsample_bytree=0.85, min_child_samples=25, reg_lambda=1.0,
                        random_state=seed, n_jobs=-1, verbose=-1)
    mdl.fit(X[tr], y[tr])
    pred = mdl.predict(X)

    gm = float(y[tr].mean())
    fam_mean = {}
    for f in fams:
        m = attempted[f].to_numpy() & tr
        fam_mean[f] = float(y[m].mean()) if m.sum() >= 10 else gm
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for f in fams:
        a = attempted[f].to_numpy()
        acc += a * fam_mean[f]; cnt += a
    amean = np.where(cnt > 0, acc / np.maximum(cnt, 1), gm)

    res = {"prospective_model": mae(pred[te], y[te]),
           "predict_zero": mae(np.zeros(te.sum()), y[te]),
           "action_mean": mae(amean[te], y[te]),
           "predict_mean": mae(np.full(te.sum(), gm), y[te])}
    return res, pred, y


def cluster_upper_bound(err: np.ndarray, groups: np.ndarray, alpha_seed: int = 0, B: int = 400) -> float:
    """Upper 95% confidence bound on the failure rate that respects organization clustering.

    Individual transitions are not exchangeable: they are clustered within organizations.
    We therefore resample whole organizations rather than rows, so the bound reflects the
    number of organizations supporting it rather than the (much larger) number of rows.
    """
    if len(err) == 0:
        return 1.0
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(alpha_seed)
    rates = np.empty(B)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        rates[b] = err[rows].mean()
    return float(np.quantile(rates, 0.95))


def reliability_stage(pairs, X, y, pred_col, tr, cal, thr, te, seed, tau=1.0,
                      groups_out=None, test_mask=None):
    """Reliability model + group-aware selective acceptance, all prospective."""
    err = (np.abs(pred_col - y) > tau).astype(int)
    pos = max(1, int(err[cal].sum())); neg = max(1, int((~err[cal].astype(bool)).sum()))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.85,
                         colsample_bytree=0.85, min_child_samples=25, scale_pos_weight=neg / pos,
                         reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(X[cal], err[cal])
    risk = clf.predict_proba(X)[:, 1]
    auc = (float(roc_auc_score(err[te], risk[te]))
           if 0 < err[te].sum() < te.sum() else None)

    groups = pairs["company_reference"].to_numpy()
    out = {}
    for alpha in (0.05, 0.10, 0.20):
        best_t, best_cov = None, 0.0
        for t in np.quantile(risk[thr], np.linspace(0.02, 1.0, 50)):
            acc = risk[thr] <= t
            if acc.sum() < 20:
                continue
            ub = cluster_upper_bound(err[thr][acc], groups[thr][acc], seed)
            if ub <= alpha and acc.mean() > best_cov:
                best_t, best_cov = float(t), float(acc.mean())
        if best_t is None:
            out[str(alpha)] = {"coverage": None, "empirical": None}
        else:
            a_te = risk[te] <= best_t
            rec = {"coverage": float(a_te.mean()),
                   "empirical": float(err[te][a_te].mean()) if a_te.any() else None,
                   "n_records": int(a_te.sum())}
            if groups_out is not None and test_mask is not None:
                rec["n_orgs"] = int(pd.unique(groups_out[test_mask][a_te]).size)
            out[str(alpha)] = rec
    return auc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--sample-rows", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--objective", default="l1", choices=["l1","l2"],
                    help="training loss; l1 matches the MAE we report and resists heavy tails")
    ap.add_argument("--cal-splits", type=int, default=5,
                    help="independent organization splits for the risk-calibration report")
    ap.add_argument("--temporal-orgs", action="store_true", help="unseen organizations AND a later period")
    ap.add_argument("--temporal", action="store_true", help="hold out the latest period instead of organizations")
    a = ap.parse_args()

    pairs, X, attempted, fams = build(a.domain, a.topk, a.sample_rows)
    y = pairs["true_gain"].to_numpy(float)
    print(f"\n########## {a.domain} (PROSPECTIVE) ##########", flush=True)
    print(f"attempted-action transitions: {len(pairs):,}  organizations: {pairs.company_reference.nunique():,}")
    print(f"features: {X.shape[1]}  mean gain={y.mean():+.3f}  median={np.median(y):+.3f}  "
          f"share |g|<0.5: {np.mean(np.abs(y) < 0.5)*100:.0f}%", flush=True)

    runs = []
    if a.temporal or a.temporal_orgs:
        cut = pairs["old_date"].quantile(0.70)
        early = (pairs["old_date"] <= cut).to_numpy()
        if a.temporal_orgs:
            # simultaneous generalization: unseen organizations AND a later period
            rng = np.random.default_rng(42)
            comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
            trc = set(comps[:int(0.70 * len(comps))])
            in_tr_org = pairs.company_reference.isin(trc).to_numpy()
            tr = early & in_tr_org; te = (~early) & (~in_tr_org)
            print(f"TEMPORAL+ORG holdout at {pd.Timestamp(cut).date()}: "
                  f"train {tr.sum():,} (earlier, train orgs) / test {te.sum():,} (later, unseen orgs)", flush=True)
        else:
            tr = early; te = ~early
            print(f"TEMPORAL holdout at {pd.Timestamp(cut).date()} (organizations may span both sides): "
                  f"train {tr.sum():,} / test {te.sum():,}", flush=True)
        r, _, _ = evaluate(pairs, X, attempted, fams, tr, te, 42, a.objective)
        runs.append(r)
    else:
        for s in range(a.seeds):
            rng = np.random.default_rng(42 + s)
            comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
            trc = set(comps[:int(0.70 * len(comps))])
            tr = pairs.company_reference.isin(trc).to_numpy(); te = ~tr
            r, _, _ = evaluate(pairs, X, attempted, fams, tr, te, 42 + s, a.objective)
            runs.append(r)
        print(f"organization-held-out, {a.seeds} seeds: "
              f"train~{tr.sum():,} test~{te.sum():,}", flush=True)

    print("\n  estimator            MAE (mean +/- sd over runs)")
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs])
        out[k] = {"mean": float(v.mean()), "sd": float(v.std())}
        star = "  <-- ours" if k == "prospective_model" else ""
        print(f"  {k:20s} {v.mean():.3f} +/- {v.std():.3f}{star}")
    best_ref = min(out[k]["mean"] for k in out if k != "prospective_model")
    verdict = "BEATS all references" if out["prospective_model"]["mean"] < best_ref else "does NOT beat the best reference"
    print(f"\n  VERDICT: prospective model {verdict} "
          f"({out['prospective_model']['mean']:.3f} vs best reference {best_ref:.3f})", flush=True)

    # reliability + organization-aware calibration, repeated over independent splits
    aucs, cal = [], {}
    for s in range(a.cal_splits):
        rng = np.random.default_rng(7 + s)
        comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
        n = len(comps); b1, b2, b3 = int(.55 * n), int(.75 * n), int(.85 * n)
        sets = [set(comps[:b1]), set(comps[b1:b2]), set(comps[b2:b3]), set(comps[b3:])]
        g_, c_, k_, t_ = [pairs.company_reference.isin(q).to_numpy() for q in sets]
        mdl = LGBMRegressor(objective=a.objective, n_estimators=700, learning_rate=0.03,
                            num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                            min_child_samples=25, reg_lambda=1.0, random_state=7 + s,
                            n_jobs=-1, verbose=-1)
        mdl.fit(X[g_], y[g_])
        auc, sel = reliability_stage(pairs, X, y, mdl.predict(X), g_, c_, k_, t_, 7 + s,
                                     groups_out=pairs["company_reference"].to_numpy(), test_mask=t_)
        if auc is not None:
            aucs.append(auc)
        for al, v in sel.items():
            cal.setdefault(al, []).append(v)

    print(f"\n  reliability AUC over {a.cal_splits} splits: "
          f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}" if aucs else "  reliability AUC: n/a")
    print("  organization-aware risk calibration (mean over splits; range in brackets):")
    cal_out = {}
    for al, vs in cal.items():
        answered = [v for v in vs if v["coverage"] is not None]
        if not answered:
            print(f"    target {float(al):.0%}: returns nothing in {len(vs)}/{len(vs)} splits")
            cal_out[al] = {"answered": 0, "n_splits": len(vs)}
            continue
        cov = np.array([v["coverage"] for v in answered])
        err = np.array([v["empirical"] for v in answered if v["empirical"] is not None])
        norg = np.array([v.get("n_orgs", np.nan) for v in answered], float)
        nrec = np.array([v.get("n_records", np.nan) for v in answered], float)
        exceed = int(sum(1 for e in err if e > float(al)))
        cal_out[al] = {"answered": len(answered), "n_splits": len(vs),
                       "coverage_mean": float(cov.mean()), "coverage_min": float(cov.min()),
                       "coverage_max": float(cov.max()), "error_mean": float(err.mean()) if len(err) else None,
                       "error_min": float(err.min()) if len(err) else None,
                       "error_max": float(err.max()) if len(err) else None,
                       "exceeded_in": exceed,
                       "accepted_orgs_mean": float(np.nanmean(norg)),
                       "accepted_records_mean": float(np.nanmean(nrec))}
        print(f"    target {float(al):.0%}: answers {len(answered)}/{len(vs)} splits  "
              f"cov={cov.mean():.2f}[{cov.min():.2f},{cov.max():.2f}]  "
              f"err={err.mean():.3f}[{err.min():.3f},{err.max():.3f}]  "
              f"exceeded {exceed}/{len(err)}  "
              f"accepted orgs~{np.nanmean(norg):.0f} records~{np.nanmean(nrec):.0f}")

    # paired organization-clustered CI for the improvement over predict-zero
    rng = np.random.default_rng(42)
    comps = pairs.company_reference.unique().copy(); rng.shuffle(comps)
    trc = set(comps[:int(0.70 * len(comps))])
    tr = pairs.company_reference.isin(trc).to_numpy(); te = ~tr
    mdl = LGBMRegressor(objective=a.objective, n_estimators=700, learning_rate=0.03, num_leaves=63,
                        subsample=0.85, colsample_bytree=0.85, min_child_samples=25,
                        reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    mdl.fit(X[tr], y[tr]); pr = mdl.predict(X)
    em, ez = np.abs(pr[te] - y[te]), np.abs(y[te])
    grp = pairs["company_reference"].to_numpy()[te]
    uniq = np.unique(grp); idx = {g_: np.flatnonzero(grp == g_) for g_ in uniq}
    rb = np.random.default_rng(0); diffs = np.empty(2000)
    for b in range(2000):
        rows = np.concatenate([idx[g_] for g_ in rb.choice(uniq, size=len(uniq), replace=True)])
        diffs[b] = ez[rows].mean() - em[rows].mean()
    lo, hi = float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))
    print(f"\n  paired improvement over predict-zero: {ez.mean()-em.mean():+.3f} "
          f"95% CI [{lo:+.3f}, {hi:+.3f}] {'excludes 0' if lo>0 else 'INCLUDES 0'}")
    out_extra = {"reliability_auc_mean": float(np.mean(aucs)) if aucs else None,
                 "reliability_auc_sd": float(np.std(aucs)) if aucs else None,
                 "risk_calibration": cal_out,
                 "paired_improvement": {"reduction": float(ez.mean()-em.mean()),
                                        "ci": [lo, hi], "significant": bool(lo > 0)}}

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "temporal_orgs" if a.temporal_orgs else ("temporal" if a.temporal else "orgheldout")
    (OUT / f"{a.domain}_{tag}.json").write_text(json.dumps(
        {"domain": a.domain, "split": tag, "objective": a.objective, "n": int(len(pairs)),
         "n_orgs": int(pairs.company_reference.nunique()),
         "mean_gain": float(y.mean()), "share_small": float(np.mean(np.abs(y) < 0.5)),
         "results": out, **out_extra, "beats_all_references": bool(out["prospective_model"]["mean"] < best_ref)}, indent=2))
    print(f"wrote {OUT / (a.domain + '_' + tag + '.json')}")


if __name__ == "__main__":
    main()
