"""All figures. Plain white, thin marks, direct labels, one blue + one orange accent."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import BLUE, FIGURES_DIR, GREY, INK, LIGHT_GREY, ORANGE  # noqa: E402

AQUA = "#1baf7a"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GREY,
            "axes.labelcolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ececea",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ data
def fig_hidden_missing(report: pd.DataFrame, out_dir=None):
    _style()
    r = report.sort_values("Hidden_missing_pct")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(r["Feature"], r["Hidden_missing_pct"], color=BLUE, height=0.55)
    for y, (pct, n) in enumerate(zip(r["Hidden_missing_pct"], r["Hidden_missing_n"])):
        ax.text(pct + 0.8, y, f"{pct:.1f}%  ({n} rows)", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(r["Hidden_missing_pct"]) * 1.35)
    ax.set_xlabel("Share of all 768 patients with a 0 recorded (impossible value = not measured)")
    ax.set_title("Hidden missing values in the PIMA dataset")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig01_hidden_missing_values.png", out_dir)


# ------------------------------------------------------------------ baseline
def fig_baseline(res: dict, out_dir=None):
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.plot(res["fpr"], res["tpr"], color=BLUE, lw=2)
    a1.plot([0, 1], [0, 1], color=LIGHT_GREY, lw=1, ls="--")
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.set_title(f"ROC curve, AUC = {res['auc']:.3f}")
    cm = res["confusion"]
    a2.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.2)
    for i in range(2):
        for j in range(2):
            a2.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16, color=INK if cm[i, j] < cm.max() * 0.6 else "white")
    a2.set_xticks([0, 1], ["Predicted: no", "Predicted: yes"])
    a2.set_yticks([0, 1], ["Actual: no", "Actual: yes"])
    a2.grid(False)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title(f"Accuracy {res['accuracy']:.1%}, sensitivity {res['recall']:.0%} (one split)")
    fig.suptitle("Baseline: the original recipe, one 80/20 split", x=0.01, y=1.05, ha="left", fontweight="bold")
    return _save(fig, "fig02_baseline_roc_confusion.png", out_dir)


def fig_split_variability(var: pd.DataFrame, slide_acc: float = 0.786, out_dir=None):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(var["accuracy"] * 100, bins=18, color=BLUE, edgecolor="white", linewidth=1)
    m, s = var["accuracy"].mean() * 100, var["accuracy"].std(ddof=1) * 100
    ax.axvline(m, color=INK, lw=1.5)
    ax.axvline(slide_acc * 100, color=ORANGE, lw=2)
    top = ax.get_ylim()[1]
    ax.text(m - 0.3, top * 0.95, f"mean {m:.1f}%", ha="right", va="top", fontsize=9)
    ax.text(slide_acc * 100 + 0.3, top * 0.82, f"slide: {slide_acc:.1%}", ha="left", va="top", fontsize=9, color=ORANGE)
    lo, hi = np.percentile(var["accuracy"] * 100, [2.5, 97.5])
    ax.set_xlabel("Test accuracy (%) of the same model on different random 80/20 splits")
    ax.set_ylabel("Number of splits")
    ax.set_title(f"One split is one draw: accuracy ranges {lo:.0f}% to {hi:.0f}% (95% of {len(var)} splits), SD {s:.1f} points")
    return _save(fig, "fig03_split_variability.png", out_dir)


def fig_forward_selection(fs: pd.DataFrame, out_dir=None):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fs["step"], fs["cv_auc"], color=BLUE, lw=2, marker="o", ms=6)
    short = {"DiabetesPedigreeFunction": "Pedigree", "BloodPressure": "BloodPress.", "SkinThickness": "SkinThick."}
    for i, (_, r) in enumerate(fs.iterrows()):
        above = i < 3 or i % 2 == 1
        ax.annotate(
            f"+{short.get(r['added'], r['added'])}",
            (r["step"], r["cv_auc"]),
            textcoords="offset points",
            xytext=(0, 10 if above else -16),
            ha="center",
            fontsize=8.5,
        )
    ax.set_xticks(fs["step"])
    ax.set_xlabel("Number of features in the model")
    ax.set_ylabel("5-fold cross-validated AUC")
    ax.set_title("Forward feature selection: glucose does most of the work")
    ax.set_ylim(fs["cv_auc"].min() - 0.02, fs["cv_auc"].max() + 0.03)
    return _save(fig, "fig04_forward_selection.png", out_dir)


def fig_regularization(curves: pd.DataFrame, out_dir=None):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for pen, color in (("L1", BLUE), ("L2", ORANGE)):
        d = curves[curves["penalty"] == pen]
        ax.plot(d["lambda"], d["cv_auc"], color=color, lw=2, label=f"{pen} ({'Lasso' if pen == 'L1' else 'Ridge'})")
    ax.set_xlabel("Regularisation strength (lambda = 1/C)")
    ax.set_ylabel("5-fold cross-validated AUC")
    ax.set_ylim(0.78, 0.86)  # a wide axis on purpose: the wiggles are in the 4th decimal
    ax.set_title("L1 and L2 regularisation barely change the AUC")
    ax.legend(loc="lower left")
    return _save(fig, "fig05_regularization.png", out_dir)


# ------------------------------------------------------------------ model comparison
def fig_model_comparison(tbl: pd.DataFrame, out_dir=None):
    _style()
    t = tbl.sort_values("auc_mean")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ypos = np.arange(len(t))
    ax.errorbar(t["auc_mean"], ypos, xerr=t["auc_sd"], fmt="o", color=BLUE, ecolor=GREY, elinewidth=1.5, capsize=3, ms=8)
    for y, (m, s) in enumerate(zip(t["auc_mean"], t["auc_sd"])):
        ax.text(m, y + 0.28, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_yticks(ypos, t["model"])
    ax.set_ylim(-0.6, len(t) - 0.2)
    ax.set_xlabel("Cross-validated ROC-AUC (mean, bars = +/- 1 SD over folds)")
    ax.set_title("Model comparison on identical folds: the models are statistically close")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig06_model_comparison.png", out_dir)


def fig_ablation(tbl: pd.DataFrame, out_dir=None):
    _style()
    t = tbl.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ypos = np.arange(len(t))
    ax.errorbar(t["auc_mean"], ypos, xerr=t["auc_sd"], fmt="o", color=BLUE, ecolor=GREY, elinewidth=1.5, capsize=3, ms=8)
    for y, m in enumerate(t["auc_mean"]):
        ax.text(m, y + 0.28, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_yticks(ypos, [v.split(" (")[0] if len(v) > 52 else v for v in t["variant"]], fontsize=10)
    ax.set_ylim(-0.6, len(t) - 0.2)
    ax.set_xlabel("Cross-validated ROC-AUC, Logistic Regression, same folds (bars = +/- 1 SD)")
    ax.set_title("What the data cleaning is worth")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig07_preprocessing_ablation.png", out_dir)


# ------------------------------------------------------------------ calibration
def fig_calibration(probs: dict, y, out_dir=None):
    from .calibration import reliability_points

    _style()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    styles = [(BLUE, "o", "-"), (ORANGE, "s", "-"), (AQUA, "^", "--")]
    for (name, p), (c, mk, ls) in zip(probs.items(), styles):
        mp, fr = reliability_points(y, p)
        ax.plot(mp, fr, color=c, marker=mk, ms=6, lw=1.8, ls=ls, label=name)
    ax.set_xlabel("Predicted probability of diabetes")
    ax.set_ylabel("Observed fraction with diabetes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability: closer to the dashed line = more honest")
    ax.legend(loc="upper left", fontsize=8.5)
    return _save(fig, "fig08_calibration.png", out_dir)


# ------------------------------------------------------------------ thresholds
def fig_threshold(tbl: pd.DataFrame, chosen: float, out_dir=None):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for col, c, lab, ls in (("sensitivity", BLUE, "Sensitivity (diabetics caught)", "-"), ("specificity", ORANGE, "Specificity (healthy cleared)", "-"), ("ppv", GREY, "Precision (flagged who are diabetic)", "--")):
        ax.plot(tbl["threshold"], tbl[col] * 100, color=c, lw=2, ls=ls, label=lab)
    ax.axvline(0.5, color=LIGHT_GREY, lw=1.5)
    ax.axvline(chosen, color=INK, lw=1.5)
    ax.text(0.505, 104, "default 0.5", fontsize=8.5, color=GREY, va="bottom")
    ax.text(chosen + 0.005, 104, f"screening {chosen:.2f}", fontsize=8.5, va="bottom")
    ax.set_xlabel("Probability threshold for calling a patient 'diabetic'")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.set_title("Threshold trade-off (out-of-fold predictions)", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5)
    return _save(fig, "fig09_threshold_tradeoff.png", out_dir)


# ------------------------------------------------------------------ conformal
def fig_conformal(y, p_oof, zone_edges: tuple, cov: pd.DataFrame, alpha: float, out_dir=None):
    _style()
    low_max, high_min = zone_edges
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    bins = np.linspace(0, 1, 26)
    a1.hist(p_oof[y == 0], bins=bins, color=BLUE, alpha=0.75, label="Not diabetic", edgecolor="white")
    a1.hist(p_oof[y == 1], bins=bins, color=ORANGE, alpha=0.75, label="Diabetic", edgecolor="white")
    lo, hi = sorted([low_max, high_min])
    a1.axvspan(0, min(high_min, low_max), color=BLUE, alpha=0.07)
    a1.axvspan(lo, hi, color=GREY, alpha=0.16)
    a1.axvspan(max(high_min, low_max), 1, color=ORANGE, alpha=0.07)
    top = a1.get_ylim()[1]
    a1.set_ylim(0, top * 1.2)
    ty = top * 1.17
    a1.text(min(high_min, low_max) / 2, ty, "Low risk", ha="center", va="top", fontsize=9)
    a1.text((lo + hi) / 2, ty, "Uncertain: lab test", ha="center", va="top", fontsize=9)
    a1.text((max(high_min, low_max) + 1) / 2, ty, "High risk", ha="center", va="top", fontsize=9)
    a1.set_xlabel("Out-of-fold predicted probability of diabetes")
    a1.set_ylabel("Patients")
    a1.set_title("Three zones from conformal calibration")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    data = [cov["coverage_class0"] * 100, cov["coverage_class1"] * 100]
    bp = a2.boxplot(data, widths=0.45, patch_artist=True, medianprops={"color": INK, "lw": 1.5}, flierprops={"marker": "o", "ms": 3, "mfc": GREY, "mec": GREY})
    for patch, c in zip(bp["boxes"], (BLUE, ORANGE)):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(GREY)
    a2.axhline((1 - alpha) * 100, color=INK, ls="--", lw=1.3)
    a2.set_xticks([1, 2], ["Not diabetic", "Diabetic"])
    a2.set_ylabel("Coverage on unseen test patients (%)")
    a2.set_title(f"Guarantee check, {len(cov)} splits\n(dashed line = {(1 - alpha) * 100:.0f}% target)", fontsize=11)
    a2.grid(axis="x", visible=False)
    return _save(fig, "fig10_conformal_zones_and_coverage.png", out_dir)


# ------------------------------------------------------------------ explanation
def fig_odds_ratios(orr: pd.DataFrame, out_dir=None):
    _style()
    t = orr.sort_values("odds_ratio_per_sd")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ypos = np.arange(len(t))
    ax.hlines(ypos, t["ci_low"], t["ci_high"], color=GREY, lw=2)
    ax.plot(t["odds_ratio_per_sd"], ypos, "o", color=BLUE, ms=8)
    ax.axvline(1.0, color=INK, lw=1.2, ls="--")
    ax.set_yticks(ypos, t["feature"])
    ax.set_xscale("log")
    ax.set_xlim(0.65, 4.6)
    ax.set_xticks([0.8, 1, 1.5, 2, 3, 4], ["0.8", "1", "1.5", "2", "3", "4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Odds ratio per +1 SD (log scale, bars = bootstrap 95% interval)")
    ax.set_title("Which features matter? Interval crossing 1 = no clear effect")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig11_odds_ratios.png", out_dir)


def fig_patient_explanation(pred, title: str, out_dir=None, name: str = "fig12_example_patient_explanation.png"):
    return _save(build_patient_explanation(pred, title), name, out_dir)


def build_patient_explanation(pred, title: str):
    """Figure object (not saved) so the app can show it directly."""
    _style()
    c = pred.contributions.sort_values("contribution")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    cols = [ORANGE if v > 0 else BLUE for v in c["contribution"]]
    ax.barh(c["feature"], c["contribution"], color=cols, height=0.55)
    for y, (v, val, miss) in enumerate(zip(c["contribution"], c["value"], c["was_missing"])):
        label = "not measured" if miss else f"{val:g}"
        ax.text(v + (0.03 if v >= 0 else -0.03), y, label, va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Push on log-odds of diabetes (orange raises risk, blue lowers it) vs the average patient")
    ax.set_title(f"{title}: P(diabetes) = {pred.probability:.0%}, {pred.decision}")
    lim = max(abs(c["contribution"]).max() * 1.35, 0.3)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    return fig


def fig_what_if(p_before: float, zone_edges: tuple, steps: list, out_dir=None):
    return _save(build_risk_ruler(p_before, zone_edges, steps), "fig13_what_if_risk_ruler.png", out_dir)


def build_risk_ruler(p_before: float, zone_edges: tuple, steps: list):
    """A risk ruler figure (not saved). `steps` = list of (label, probability)."""
    _style()
    low_max, high_min = zone_edges
    lo, hi = sorted([low_max, high_min])
    fig, ax = plt.subplots(figsize=(9, 2.7))
    ax.axvspan(0, min(low_max, high_min), color=BLUE, alpha=0.12)
    ax.axvspan(lo, hi, color=GREY, alpha=0.2)
    ax.axvspan(max(low_max, high_min), 1, color=ORANGE, alpha=0.12)
    ax.text(min(low_max, high_min) / 2, 1.05, "Low risk", ha="center", fontsize=9)
    ax.text((lo + hi) / 2, 1.05, "Uncertain", ha="center", fontsize=9)
    ax.text((max(low_max, high_min) + 1) / 2, 1.05, "High risk", ha="center", fontsize=9)
    pts = [("Now", p_before)] + steps
    for i, (lab, p) in enumerate(pts):
        ax.plot([p], [0.5], "o", ms=10, color=INK if i == 0 else BLUE, zorder=3)
        ax.annotate(f"{lab}\n{p:.0%}", (p, 0.5), textcoords="offset points", xytext=(0, 14 if i % 2 == 0 else -34), ha="center", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Predicted probability of diabetes")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Where this patient sits (and where the what-if change would move them)", pad=18)
    return fig


def fig_subgroups(tbl: pd.DataFrame, out_dir=None):
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.arange(len(tbl))
    a1.bar(x, tbl["auc"], color=BLUE, width=0.55)
    for i, v in enumerate(tbl["auc"]):
        a1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    a1.set_ylim(0.5, 1.0)
    a1.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(tbl["age_group"], tbl["n"])])
    a1.set_ylabel("Out-of-fold AUC")
    a1.set_title("Ranking quality by age group")
    a1.grid(axis="x", visible=False)
    w = 0.38
    a2.bar(x - w / 2, tbl["sensitivity_pct"], w, color=BLUE, label="Sensitivity")
    a2.bar(x + w / 2, tbl["specificity_pct"], w, color=ORANGE, label="Specificity")
    a2.set_xticks(x, tbl["age_group"])
    a2.set_ylabel("% at threshold 0.5")
    a2.set_ylim(0, 105)
    a2.set_title("Error pattern by age group")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5)
    a2.grid(axis="x", visible=False)
    return _save(fig, "fig14_age_subgroups.png", out_dir)
