"""Run the whole project: baseline, extension, tables, figures.

    python scripts/run_all.py            # full run (a few minutes)
    python scripts/run_all.py --quick    # fewer repeats, for a fast check

Outputs go to results/figures, results/tables and results/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from diabetes_ml import plots  # noqa: E402
from diabetes_ml.baseline import forward_selection, regularization_curves, run_baseline, split_variability  # noqa: E402
from diabetes_ml.calibration import calibration_study  # noqa: E402
from diabetes_ml.conformal import evaluate_coverage  # noqa: E402
from diabetes_ml.config import ALPHA, FEATURES, FIGURES_DIR, RANDOM_STATE, RESULTS_DIR, TABLES_DIR, TARGET  # noqa: E402
from diabetes_ml.data import hidden_missing_report, load_pima  # noqa: E402
from diabetes_ml.evaluation import compare_models, oof_probabilities, preprocessing_ablation, subgroup_report  # noqa: E402
from diabetes_ml.explain import global_importance, odds_ratios  # noqa: E402
from diabetes_ml.preprocessing import build_pipeline  # noqa: E402
from diabetes_ml.screening import ScreeningModel  # noqa: E402
from diabetes_ml.thresholds import confusion_at, threshold_for_sensitivity, threshold_table  # noqa: E402

SLIDE_ACCURACY = 0.786  # accuracy reported on the course slides (single split)


def step(msg: str) -> float:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return time.time()


def main(quick: bool = False) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    n_split_var = 40 if quick else 200
    n_cv_rep = 2 if quick else 5
    n_cov_rep = 20 if quick else 100
    n_boot = 100 if quick else 500

    df = load_pima()
    X, y = df[FEATURES], df[TARGET].to_numpy()
    metrics: dict = {"n_patients": int(len(df)), "prevalence": float(y.mean()), "quick_run": quick}

    # ------------------------------------------------------------ 1. data
    step("1/9 Data: hidden missing values")
    rep = hidden_missing_report(df)
    rep.to_csv(TABLES_DIR / "01_hidden_missing_values.csv", index=False)
    plots.fig_hidden_missing(rep)

    # ------------------------------------------------------------ 2. baseline
    step("2/9 Baseline: the original recipe")
    base = run_baseline(df, random_state=RANDOM_STATE)
    metrics["baseline_seed42"] = {k: (float(base[k]) if np.isscalar(base[k]) else None) for k in ("accuracy", "auc", "precision", "recall")}
    metrics["baseline_seed42"]["confusion"] = base["confusion"].tolist()
    plots.fig_baseline(base)
    var = split_variability(df, n_splits=n_split_var)
    var.to_csv(TABLES_DIR / "02_baseline_split_variability.csv", index=False)
    plots.fig_split_variability(var, SLIDE_ACCURACY)
    lo, hi = np.percentile(var["accuracy"], [2.5, 97.5])
    metrics["split_variability"] = {
        "n_splits": n_split_var,
        "accuracy_mean": float(var["accuracy"].mean()),
        "accuracy_sd": float(var["accuracy"].std(ddof=1)),
        "accuracy_p2_5": float(lo),
        "accuracy_p97_5": float(hi),
        "recall_mean": float(var["recall"].mean()),
        "auc_mean": float(var["auc"].mean()),
    }
    fs = forward_selection(df)
    fs.to_csv(TABLES_DIR / "03_forward_selection.csv", index=False)
    plots.fig_forward_selection(fs)
    curves = regularization_curves(df)
    curves.to_csv(TABLES_DIR / "04_regularization_curves.csv", index=False)
    plots.fig_regularization(curves)

    # ------------------------------------------------------------ 3. comparison
    step("3/9 Fair model comparison (repeated cross-validation)")
    comp = compare_models(df, n_repeats=n_cv_rep)
    comp.to_csv(TABLES_DIR / "05_model_comparison.csv", index=False)
    plots.fig_model_comparison(comp)
    metrics["model_comparison"] = comp.round(4).to_dict(orient="records")

    step("4/9 Preprocessing ablation")
    abl = preprocessing_ablation(df, n_repeats=n_cv_rep)
    abl.to_csv(TABLES_DIR / "06_preprocessing_ablation.csv", index=False)
    plots.fig_ablation(abl)
    metrics["ablation"] = abl.round(4).to_dict(orient="records")

    # ------------------------------------------------------------ 4. calibration
    step("5/9 Calibration")
    cal_tbl, probs = calibration_study(df)
    cal_tbl.to_csv(TABLES_DIR / "07_calibration.csv", index=False)
    plots.fig_calibration(probs, y)
    metrics["calibration"] = cal_tbl.round(4).to_dict(orient="records")

    # ------------------------------------------------------------ 5. thresholds
    step("6/9 Screening threshold")
    pipe = build_pipeline()
    p_oof = oof_probabilities(pipe, X, y)
    tt = threshold_table(y, p_oof)
    tt.to_csv(TABLES_DIR / "08_threshold_table.csv", index=False)
    chosen = threshold_for_sensitivity(y, p_oof, target=0.85)
    plots.fig_threshold(tt, chosen)
    metrics["threshold"] = {"default_0.5": confusion_at(y, p_oof, 0.5), f"screening_{chosen:.2f}": confusion_at(y, p_oof, chosen), "chosen": chosen}

    # ------------------------------------------------------------ 6. conformal
    step("7/9 Conformal prediction (this is the slow step)")
    model = ScreeningModel(alpha=ALPHA).fit(df)
    cov = evaluate_coverage(df, build_pipeline(), alpha=ALPHA, n_repeats=n_cov_rep)
    cov.to_csv(TABLES_DIR / "09_conformal_coverage_repeats.csv", index=False)
    plots.fig_conformal(y, p_oof, model.zone_edges(), cov, ALPHA)
    metrics["conformal"] = {
        "alpha": ALPHA,
        "n_repeats": n_cov_rep,
        "zone_edges_low_below_high_above": [float(v) for v in model.zone_edges()],
        "coverage_class0_mean": float(cov["coverage_class0"].mean()),
        "coverage_class0_min": float(cov["coverage_class0"].min()),
        "coverage_class1_mean": float(cov["coverage_class1"].mean()),
        "coverage_class1_min": float(cov["coverage_class1"].min()),
        "pct_uncertain_mean": float(cov["pct_uncertain"].mean()),
        "pct_low_mean": float(cov["pct_low"].mean()),
        "pct_high_mean": float(cov["pct_high"].mean()),
        "accuracy_when_committed_mean": float(cov["accuracy_when_committed"].mean()),
        "pct_flagged_followup_mean": float(cov["pct_flagged_for_followup"].mean()),
        "ppv_of_flagged_mean": float(cov["ppv_of_flagged"].mean()),
    }

    # ------------------------------------------------------------ 7. explanation
    step("8/9 Explanations and what-if")
    orr = odds_ratios(df, n_boot=n_boot)
    orr.to_csv(TABLES_DIR / "10_odds_ratios.csv", index=False)
    plots.fig_odds_ratios(orr)
    gi = global_importance(model.pipeline_, X)
    gi.to_csv(TABLES_DIR / "11_global_importance.csv", index=False)

    example = df.iloc[0][FEATURES].to_dict()
    pred = model.predict(example)
    plots.fig_patient_explanation(pred, "Example patient (row 1 of the dataset)")
    wi_unc = model.what_if(example, "uncertain")
    wi_low = model.what_if(example, "low")
    plots.fig_what_if(
        pred.probability,
        model.zone_edges(),
        [("Leave 'High risk'", wi_unc["p_after"]), ("Reach 'Low risk'", wi_low["p_after"])],
    )
    metrics["example_patient"] = {
        "input": {k: float(v) for k, v in example.items()},
        "probability": pred.probability,
        "decision": pred.decision,
        "what_if_uncertain": {k: wi_unc[k] for k in ("feasible", "p_after", "changes")},
        "what_if_low": {k: wi_low[k] for k in ("feasible", "p_after", "changes")},
    }

    # ------------------------------------------------------------ 8. subgroups
    step("9/9 Age subgroups")
    sub = subgroup_report(df, p_oof)
    sub.to_csv(TABLES_DIR / "12_age_subgroups.csv", index=False)
    plots.fig_subgroups(sub)

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    step(f"Done. Figures: {FIGURES_DIR}  Tables: {TABLES_DIR}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer repeats for a fast smoke run")
    main(ap.parse_args().quick)
