"""Central configuration: column names, constants and the "what-if" limits."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# In the PIMA file a 0 in these columns is physiologically impossible. It means
# "not measured", so it must be treated as missing, not as a real value.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Threshold the original project used to call an insulin value "extreme".
ORIGINAL_INSULIN_CAP = 425

RANDOM_STATE = 42

# Conformal prediction: target miss rate per class (0.10 -> at least 90 % coverage).
ALPHA = 0.10

# "What-if" analysis. Only these features may be lowered, and only down to the
# floor shown (a healthy lower bound), never below it.
MODIFIABLE_FLOORS = {
    "Glucose": 70.0,
    "BMI": 18.5,
    "BloodPressure": 60.0,
}

# Plausible entry ranges used for input validation in the app and API.
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 300),
    "BloodPressure": (30, 140),
    "SkinThickness": (5, 100),
    "Insulin": (10, 900),
    "BMI": (10, 70),
    "DiabetesPedigreeFunction": (0.05, 2.5),
    "Age": (21, 90),
}

# Plot colours (first two slots of a colour-blind-validated palette, plus greys).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#1a1a19"
GREY = "#8a8985"
LIGHT_GREY = "#d9d8d4"
