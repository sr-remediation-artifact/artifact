# Artifact: What Remediation Actually Buys

Projected versus realized gains in commercial security ratings.

This artifact contains the complete pipeline, the exact configuration behind every number
in the paper, the aggregate result files those numbers are read from, and a synthetic
data generator so the pipeline can be executed end to end without the proprietary data.

## What is and is not here

**Here.** All analysis code, all result JSONs, and a generator that produces tables with
the same schema and longitudinal structure as the real ones.

**Not here.** The underlying observations. They describe externally observable
configuration data for real, identifiable organizations, collected by a commercial rating
platform under a research agreement. Organizations remain re-identifiable from raw
configuration content even with vendor-assigned identifiers removed, so the data cannot
be released. The vendor has approved publication of the aggregate results reported in the
paper, which are the JSON files under `results/`.

The synthetic data is **not** drawn from the real distribution and will **not** reproduce
the paper's numbers. It exists so a reviewer can confirm the code runs, that splits are
organization-disjoint, and that each statistic is computed as described.

## Layout

```
make_synthetic.py                 generate stand-in tables with the real schema
scripts/prospective_impact.py     RQ1 and RQ2: prospective estimator, reliability,
                                  organization-clustered calibration, temporal holdouts
scripts/model_class_comparison.py five learners on the RQ1 protocol (Table 5): three
                                  gradient-boosting libraries, Ridge, and an MLP
scripts/attempt_label_validation.py  internal validity: asset-loss confound, event-threshold
                                  sensitivity, paired organization-bootstrap CI, ablation
scripts/decision_value.py         RQ3: budget-constrained ranking against perfect foresight
scripts/per_action_and_risk_curve.py  per-action prospective accuracy, risk-coverage curve
scripts/prospective_vs_rules_oracle.py  RQ1: the rule-based counterfactual, prospectively
scripts/cross_domain_reliability.py     shared domain definitions and feature selection
results/                          the JSON files every reported number is read from
results/rules_counterfactual/     aggregates for the numbers whose per-record source
                                  cannot be released: Table 9 per action, the score
                                  reconstruction fidelity, the action-type bundling shares,
                                  and the counterfactual projected over multi-action bundles
```

## Numbers whose per-record source cannot be released

Four reported quantities are computed from material the agreement does not let us publish:
the per-record counterfactual predictions carry organization identifiers, and the score
reconstruction runs inside `web_rules_scoring_engine.py`, which encodes vendor scoring
material. `results/rules_counterfactual/` therefore holds the aggregates themselves, with
identifiers, file paths and vendor schema names removed:

| File | Supports |
|---|---|
| `per_action_counterfactual.json` | Table 9 and Section 6.1, the rule-based counterfactual against realized outcomes, per action type and in total |
| `score_reconstruction_fidelity.json` | the score-reconstruction validation in Section 5.3: MAE 3.1e-5 and exact agreement on 99.8% of 1,000 records |
| `family_bundling.json` | the one-action-type and three-or-more shares per domain, and the event counts of Table 2 |
| `bundle_counterfactual.json` | the multi-action counterfactual and robustness results in Sections 6.1 and 6.4 |

The counterfactual is evaluated on two populations drawn from the fixed test split of 23,098 Web
snapshots. `per_action_counterfactual.json` covers records in which one action type moved, where
the projection and the realized gain both belong to that action type.
`bundle_counterfactual.json` covers records in which several moved
together, where both sides describe the whole bundle: every action type in the bundle is applied
to the pre-remediation snapshot and iterated to a fixed point, so the projection describes the
state in which all of the work is complete and does not depend on the order of application.
That file records the order check, 273 records reprojected with the bundle reversed and a
maximum difference of zero, alongside an organization-clustered bootstrap interval.

## Reproducing the reported numbers

The result files under `results/` are the outputs of the commands below, run against the
proprietary tables. Each number in the paper is read from these files rather than
transcribed, so they can be checked directly against the text.

```bash
# RQ1 and RQ2, per domain
python3 scripts/prospective_impact.py --domain web            # 5 organization splits
python3 scripts/prospective_impact.py --domain web --temporal
python3 scripts/prospective_impact.py --domain web --temporal-orgs

# internal validity of the remediation-event label (Section 6.4)
python3 scripts/attempt_label_validation.py --domain web

# RQ3, decision value under a budget
python3 scripts/decision_value.py --domain web

# per-action accuracy and the risk-coverage curve
python3 scripts/per_action_and_risk_curve.py --domain web

# RQ1, the rule-based counterfactual
python3 scripts/prospective_vs_rules_oracle.py

# model class comparison (Table 5): same protocol, only the regressor changes
python3 scripts/model_class_comparison.py --domain web
```

Replace `web` with `mail` or `vulnerability` for the other domains.

## Running against synthetic data

```bash
python3 make_synthetic.py --out data/processed
python3 scripts/prospective_impact.py --domain web
```

Expect the pipeline to complete and print the same statistics in the same format. The
values will differ from the paper because the inputs are synthetic.

This doubles as a null check. The generator draws score movement independently of the
remediation headroom, so there is no relationship to find. Running all three domains on the
synthetic tables reproduces exactly that:

| Domain        | Ours  | Predict zero | Best reference | AUC   | Paired vs predict zero | Calibration |
|---------------|-------|--------------|----------------|-------|------------------------|-------------|
| Web           | 1.234 | 1.224        | 1.206          | 0.507 | -0.003 [-0.038, +0.032] | returns nothing, 5/5 splits |
| Mail          | 1.290 | 1.282        | 1.250          | 0.513 | -0.058 [-0.116, +0.007] | returns nothing, 5/5 splits |
| Vulnerability | 1.212 | 1.198        | 1.190          | 0.498 | -0.033 [-0.078, +0.017] | returns nothing, 5/5 splits |

MAE in score points, mean over five organization splits. In every domain the estimator
fails to beat predicting no change, every paired interval covers zero, the reliability
signal sits at chance, and the calibration criterion declines to answer at all three
targets. A pipeline that manufactured signal would not behave that way.

Scripts locate their inputs by walking up from `scripts/` to the directory holding
`make_synthetic.py`, so the commands above work in an unpacked artifact with no git
metadata present. Set `SR_DATA_ROOT` to override that and point at another directory
containing `data/processed/`.

## Demo

`demo/index.html` is a self-contained page (no build step, no network). Open it directly in
a browser. It walks through one held-out organization at a time: the remediation options
open to them, what the estimator would have told them beforehand, and then, on request,
what the score actually did and what the platform's rules had claimed.

Everything it shows is exported from the pipeline, by `scripts/export_case_files.py` into
`demo/case_files.json` (the per-organization walkthrough) and `scripts/export_demo_data.py`
into `demo/demo_data.json` (the aggregate tables). Nothing on the page is hand-drawn or
hand-typed.

Ten of the fifteen organizations were chosen to show the contrast with the rule-based
counterfactual and five were drawn at random from the remainder; the page labels which is
which, and re-running `export_case_files.py` regenerates both groups.

## Requirements

Python 3.10+, with `numpy`, `pandas`, `scikit-learn`, `lightgbm`, and `xgboost`.

## Determinism

Every script takes a fixed seed and reports the seed it used. Accuracy is averaged over
five organization splits (seeds 42 to 46); the calibration report uses seeds 7 to 11.
Reruns on the same inputs reproduce the published JSONs exactly.
