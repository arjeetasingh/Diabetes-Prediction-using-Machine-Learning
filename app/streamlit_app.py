"""Streamlit demo: diabetes risk screening with uncertainty, explanation and what-if.

Run from the project folder:
    streamlit run app/streamlit_app.py

Educational demo only. Not a medical device and not for clinical decisions.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from diabetes_ml import plots  # noqa: E402
from diabetes_ml.config import MODIFIABLE_FLOORS, VALID_RANGES, ZERO_MEANS_MISSING  # noqa: E402
from diabetes_ml.counterfactual import changes_table  # noqa: E402
from diabetes_ml.data import load_pima  # noqa: E402
from diabetes_ml.screening import ScreeningModel  # noqa: E402

st.set_page_config(page_title="Diabetes screening demo", layout="wide")

DEFAULTS = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.40,
    "Age": 35,
}
HELP = {
    "Pregnancies": "Number of times pregnant.",
    "Glucose": "Plasma glucose 2 hours into an oral glucose tolerance test (mg/dL).",
    "BloodPressure": "Diastolic blood pressure (mm Hg).",
    "SkinThickness": "Triceps skin-fold thickness (mm).",
    "Insulin": "2-hour serum insulin (mu U/ml).",
    "BMI": "Body mass index (kg/m2).",
    "DiabetesPedigreeFunction": "Score of diabetes family history (higher = more family history).",
    "Age": "Years (the dataset covers women aged 21 and over).",
}
SAVE_KEY = "result"


@st.cache_resource(show_spinner="Training the model (takes a second)...")
def get_model() -> ScreeningModel:
    return ScreeningModel().fit(load_pima())


def read_inputs() -> dict:
    record = {}
    cols = st.columns(4)
    for i, feat in enumerate(DEFAULTS):
        lo, hi = VALID_RANGES[feat]
        with cols[i % 4]:
            if feat in ZERO_MEANS_MISSING:
                unknown = st.checkbox(f"{feat}: not measured", key=f"na_{feat}", help="Tick if this value is not available.")
            else:
                unknown = False
                st.caption(f"{feat}: required")
            is_int = feat in ("Pregnancies", "Age", "Glucose", "BloodPressure", "SkinThickness", "Insulin")
            default = DEFAULTS[feat]
            val = st.number_input(
                feat,
                min_value=float(lo) if not is_int else int(lo),
                max_value=float(hi) if not is_int else int(hi),
                value=default if is_int else float(default),
                step=1 if is_int else (0.1 if feat == "BMI" else 0.01),
                help=HELP[feat],
                disabled=unknown,
                key=f"in_{feat}",
            )
        record[feat] = None if unknown else val
    return record


def main() -> None:
    model = get_model()
    low_below, high_above = model.zone_edges()

    st.title("Diabetes risk screening: with uncertainty, explanation and what-if")
    st.caption(
        "Educational demo built on the PIMA Indians dataset (768 women, Pima heritage, age 21+). "
        "Not a medical device. A result here must never replace a lab test or a doctor."
    )

    st.subheader("Patient values")
    record = read_inputs()

    if not st.button("Run screening", type="primary"):
        st.info("Enter the values above and press **Run screening**.")
        return

    try:
        pred = model.predict(record)
    except ValueError as err:
        st.error(str(err))
        return

    st.divider()
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader(f"Result: {pred.decision}")
        st.metric("Predicted probability of diabetes", f"{pred.probability:.0%}")
        st.write(pred.advice)
        if pred.not_measured:
            st.warning("Not measured (filled with the typical value): " + ", ".join(pred.not_measured))
        st.caption(
            f"How to read it: below {low_below:.0%} the answer is 'Low risk', above {high_above:.0%} it is "
            f"'High risk', and in between the model says 'Uncertain'. Those cut-offs come from conformal "
            f"calibration (90% of diabetic and 90% of non-diabetic patients in the data fall on the right side). "
            f"'Uncertain' is a feature: it is how the model avoids guessing."
        )
    with right:
        fig = plots.build_patient_explanation(pred, "Why")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("What would have to change? (what-if)")
    st.caption(
        "The smallest reduction in glucose, BMI and blood pressure (never below healthy floors "
        + ", ".join(f"{k} {v:g}" for k, v in MODIFIABLE_FLOORS.items())
        + ") that moves the model's answer. This describes the MODEL's learned pattern. It is not a promise "
        "that changing a number would change a real person's health."
    )
    goal_label = st.radio("Goal", ["Stop being 'High risk'", "Reach 'Low risk'"], horizontal=True)
    goal = "uncertain" if goal_label.startswith("Stop") else "low"
    wi = model.what_if(record, goal)
    if wi["already_met"]:
        st.success("This profile already meets the goal. No change needed.")
    elif not wi["changes"]:
        st.info("None of glucose, BMI or blood pressure can be lowered further (or they were not measured).")
    else:
        st.dataframe(changes_table(wi).round(1), hide_index=True)
        if wi["feasible"]:
            st.write(f"After these changes the model's probability would be **{wi['p_after']:.0%}**.")
        else:
            st.warning(
                f"Even with all three at their healthy floors the probability only drops to {wi['p_after']:.0%}. "
                "Other factors (age, pregnancies, family history) keep the risk high."
            )
        fig2 = plots.build_risk_ruler(pred.probability, (low_below, high_above), [("After change", wi["p_after"])])
        st.pyplot(fig2)
        plt.close(fig2)


main()
