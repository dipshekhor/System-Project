"""
personalized_nutrition.py
=========================
Personalized Nutrition Engine
------------------------------
Predicts a user's PERSONAL daily nutrient targets (calories, protein, carbs,
fat, fiber, sugar, sodium, cholesterol) from their physical profile + diseases,
then evaluates a food against those targets using a "budget impact" model.

Pipeline:
  1. Train  → train_and_save() reads 3 CSVs (Personalized_Diet_Recommendations,
              detailed_meals_macros_CLEANED, Food_and_Nutrition__) and learns
              a MultiOutputRegressor(RandomForestRegressor) → user_target_model.pkl
  2. Load   → load() reads the .pkl + feature columns at FastAPI startup
  3. Use    → predict_user_targets(profile) → 8 targets
              check_personalized_verdict(food_name, profile) →
                {targets, budget_impact, verdict, reasons}

Why a multi-output regressor?
  Each user's "daily limit" depends on their entire profile (age, weight,
  activity, multiple concurrent diseases). Static thresholds in
  medical_rules.py answer "is this food safe at all?" — this module answers
  "how much of YOUR specific daily limit does this food consume?".
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
_THIS_DIR   = Path(__file__).parent
_DATA_DIR   = _THIS_DIR.parent / "data"
_MODELS_DIR = _THIS_DIR.parent / "ml_models"

MODEL_PATH    = _MODELS_DIR / "user_target_model.pkl"
FEATURES_PATH = _MODELS_DIR / "user_target_features.json"
TARGETS_PATH  = _MODELS_DIR / "user_target_columns.json"

# ─── Schema constants ─────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "age", "gender", "height", "weight", "bmi", "activity_level",
    "is_diabetes", "is_hypertension", "is_heart",
    "is_weight_gain", "is_kidney",
]

TARGET_COLS: list[str] = [
    "target_calories", "target_protein", "target_carbs", "target_fat",
    "target_fiber", "target_sugar", "target_sodium", "target_cholesterol",
]

# Gender encoding (model needs numeric features)
_GENDER_MAP = {"male": 0, "m": 0, "female": 1, "f": 1, "other": 2}

# Activity level encoding 1..5
_ACTIVITY_MAP = {
    "sedentary": 1,
    "lightly active": 2,
    "lightly_active": 2,
    "moderately active": 3,
    "moderately_active": 3,
    "very active": 4,
    "very_active": 4,
    "extremely active": 5,
    "extra active": 5,
    "extremely_active": 5,
}

# Module-level singletons (loaded once at startup)
_model: object | None = None
_feature_names: list[str] | None = None
_target_names: list[str] | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────────────────────────────

def _encode_gender(value) -> int:
    if value is None:
        return 2
    return _GENDER_MAP.get(str(value).strip().lower(), 2)


def _encode_activity(value) -> int:
    if value is None:
        return 3                     # default Moderately Active
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Already numeric — clamp to 1..5
        return max(1, min(5, int(value)))
    return _ACTIVITY_MAP.get(str(value).strip().lower(), 3)


def _exercise_freq_to_activity(freq) -> int:
    """Map PDR Exercise_Frequency (0..6) → activity_level (1..5)."""
    try:
        f = int(freq)
    except (TypeError, ValueError):
        return 3
    if f <= 0:    return 1
    if f <= 2:    return 2
    if f <= 4:    return 3
    if f == 5:    return 4
    return 5


def _parse_disease_string(val) -> dict[str, int]:
    """
    Parse a Disease/Chronic_Disease cell into 5 binary flags.
    Treats Obesity as is_weight_gain too (clinical synonym in user mapping).
    """
    flags = {
        "is_diabetes": 0, "is_hypertension": 0, "is_heart": 0,
        "is_weight_gain": 0, "is_kidney": 0,
    }
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return flags
    text = str(val).lower()
    if text in ("none", "nan", ""):
        return flags
    if "diabetes" in text:                      flags["is_diabetes"] = 1
    if "hypertension" in text:                  flags["is_hypertension"] = 1
    if "heart" in text:                         flags["is_heart"] = 1
    if "weight gain" in text or "obesity" in text:
        flags["is_weight_gain"] = 1
    if "kidney" in text:                        flags["is_kidney"] = 1
    return flags


def _build_feature_row(profile: dict) -> dict:
    """
    Build a feature row in the exact order/encoding the model expects.
    Accepts loose user_profile dicts (height_cm, weight_kg, diseases list, etc.)
    and normalizes them to the 11 FEATURE_COLS.
    """
    height = profile.get("height") or profile.get("height_cm") or 0
    weight = profile.get("weight") or profile.get("weight_kg") or 0
    bmi = profile.get("bmi")
    if (not bmi) and height and weight:
        try:
            bmi = round(float(weight) / ((float(height) / 100) ** 2), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            bmi = 0

    diseases_list = profile.get("diseases") or []
    disease_flags = _parse_disease_string(", ".join(diseases_list))
    # Allow caller to pre-supply is_* flags directly — overrides the parsed ones
    for k in ("is_diabetes", "is_hypertension", "is_heart",
              "is_weight_gain", "is_kidney"):
        if k in profile:
            disease_flags[k] = int(profile[k])

    return {
        "age":            float(profile.get("age", 30) or 30),
        "gender":         _encode_gender(profile.get("gender")),
        "height":         float(height or 0),
        "weight":         float(weight or 0),
        "bmi":            float(bmi or 0),
        "activity_level": _encode_activity(profile.get("activity_level")),
        **disease_flags,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def _clinical_sodium(flags: pd.DataFrame) -> pd.Series:
    """
    Disease-driven sodium target in mg.
      - Hypertension / Heart Disease / Kidney Disease  → 1500 mg (AHA/NKF strict)
      - Otherwise                                      → 2300 mg (FDA general)

    Used for all 3 datasets so the model never mixes mg with the meals CSV's
    mystery-unit sodium values (10-65 range, not in mg).
    """
    risky = (flags["is_hypertension"] == 1) | (flags["is_heart"] == 1) | (flags["is_kidney"] == 1)
    return pd.Series(np.where(risky, 1500.0, 2300.0), index=flags.index)


def _clinical_sugar(flags: pd.DataFrame) -> pd.Series:
    """
    Disease-driven added-sugar target in g.
      - Diabetes  → 15 g (ADA strict)
      - Otherwise → 25 g (WHO upper limit for added sugar)

    The meals CSV `Sugar` column reports total dietary sugar (60-218 g) which
    is on a different definition than added sugar — using it as a regression
    target produced clinically wrong recommendations.
    """
    return pd.Series(np.where(flags["is_diabetes"] == 1, 15.0, 25.0), index=flags.index)


def _load_pdr() -> pd.DataFrame:
    """Personalized_Diet_Recommendations.csv → standardized feature/target frame."""
    df = pd.read_csv(_DATA_DIR / "Personalized_Diet_Recommendations.csv")
    out = pd.DataFrame()
    out["age"]            = df["Age"]
    out["gender"]         = df["Gender"].map(_encode_gender)
    out["height"]         = df["Height_cm"]
    out["weight"]         = df["Weight_kg"]
    out["bmi"]            = df["BMI"]
    out["activity_level"] = df["Exercise_Frequency"].map(_exercise_freq_to_activity)

    flags = df["Chronic_Disease"].apply(_parse_disease_string).apply(pd.Series)
    out = pd.concat([out, flags], axis=1)

    # Macros (calories, protein, carbs, fat, fiber) come from meals CSV only.
    # PDR recommendations conflict with meals labels → set to NaN so per-target
    # training skips PDR rows for these columns.
    out["target_calories"] = np.nan
    out["target_protein"]  = np.nan
    out["target_carbs"]    = np.nan
    out["target_fat"]      = np.nan
    out["target_fiber"]    = np.nan
    # Sodium and sugar targets are disease-driven clinical heuristics
    out["target_sodium"] = _clinical_sodium(out)
    out["target_sugar"]  = _clinical_sugar(out)
    return out


def _load_meals() -> pd.DataFrame:
    """detailed_meals_macros_CLEANED.csv → standardized feature/target frame."""
    df = pd.read_csv(_DATA_DIR / "detailed_meals_macros_CLEANED.csv")
    out = pd.DataFrame()
    out["age"]            = df["Ages"]
    out["gender"]         = df["Gender"].map(_encode_gender)
    out["height"]         = df["Height"]
    out["weight"]         = df["Weight"]
    out["bmi"]            = (df["Weight"] / ((df["Height"] / 100) ** 2)).round(1)
    out["activity_level"] = df["Activity Level"].map(_encode_activity)

    flags = df["Disease"].apply(_parse_disease_string).apply(pd.Series)
    out = pd.concat([out, flags], axis=1)

    out["target_calories"] = df["Daily Calorie Target"]
    out["target_protein"]  = df["Protein"]
    out["target_carbs"]    = df["Carbohydrates"]
    out["target_fat"]      = df["Fat"]
    out["target_fiber"]    = df["Fiber"]
    # Sodium/sugar use clinical heuristics, NOT meals CSV values.
    # Meals CSV sodium is in a non-mg unit (10-65 range) and meals CSV sugar
    # represents total dietary sugar, not added sugar — both unsafe to learn.
    out["target_sodium"]   = _clinical_sodium(out)
    out["target_sugar"]    = _clinical_sugar(out)
    return out


def _load_fan() -> pd.DataFrame:
    """Food_and_Nutrition__.csv → standardized feature/target frame."""
    df = pd.read_csv(_DATA_DIR / "Food_and_Nutrition__.csv")
    out = pd.DataFrame()
    out["age"]            = df["Ages"]
    out["gender"]         = df["Gender"].map(_encode_gender)
    out["height"]         = df["Height"]
    out["weight"]         = df["Weight"]
    out["bmi"]            = (df["Weight"] / ((df["Height"] / 100) ** 2)).round(1)
    out["activity_level"] = df["Activity Level"].map(_encode_activity)

    flags = df["Disease"].apply(_parse_disease_string).apply(pd.Series)
    out = pd.concat([out, flags], axis=1)

    out["target_calories"] = df["Daily Calorie Target"]
    out["target_protein"]  = df["Protein"]
    out["target_carbs"]    = df["Carbohydrates"]
    out["target_fat"]      = df["Fat"]
    out["target_fiber"]    = df["Fiber"]
    out["target_sodium"]   = _clinical_sodium(out)
    out["target_sugar"]    = _clinical_sugar(out)
    return out


def _build_training_frame() -> pd.DataFrame:
    """Merge the three sources, add synthetic cholesterol target, dedupe & clean."""
    pdr   = _load_pdr()
    meals = _load_meals()
    fan   = _load_fan()
    df = pd.concat([pdr, meals, fan], ignore_index=True)

    # Synthetic cholesterol target — stricter cap for cardio/metabolic patients
    risky = (df["is_heart"] == 1) | (df["is_hypertension"] == 1) | (df["is_diabetes"] == 1)
    df["target_cholesterol"] = np.where(risky, 200.0, 300.0)

    # Drop rows missing core demographic features only.
    # Target columns may be NaN for PDR rows (macros) — handled per-target in training.
    df = df.dropna(subset=FEATURE_COLS)

    # Sanitize demographics — invalid physiology would corrupt every target model.
    df = df[(df["age"].between(1, 120)) &
            (df["height"].between(80, 230)) &
            (df["weight"].between(20, 250))]

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save() -> dict:
    """
    Train one RandomForestRegressor per target column.
    Each model is trained only on rows where that target is not NaN:
      - Macros (calories/protein/carbs/fat/fiber): meals + FaN rows only (~3,400)
      - Sugar / sodium / cholesterol:              all rows (~8,400)
    Saves a dict {target_name: fitted_model} to user_target_model.pkl.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    print("=" * 60)
    print("Training Personalized Nutrition Engine (per-target)")
    print("=" * 60)

    df = _build_training_frame()
    print(f"  Total rows after merging & cleaning: {len(df)}")
    print(f"  Disease prevalence:")
    for col in ["is_diabetes", "is_hypertension", "is_heart",
                "is_weight_gain", "is_kidney"]:
        print(f"    {col:18s}: {int(df[col].sum()):4d}")

    X_all = df[FEATURE_COLS].to_numpy()

    models: dict[str, RandomForestRegressor] = {}
    metrics: dict[str, dict] = {}

    print("\n  Per-target training:")
    for target in TARGET_COLS:
        valid_mask = df[target].notna().to_numpy()
        n_valid = int(valid_mask.sum())
        X = X_all[valid_mask]
        y = df[target].to_numpy()[valid_mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))
        metrics[target] = {"mae": round(mae, 2), "r2": round(r2, 3),
                           "n_rows": n_valid}
        print(f"    {target:22s}  rows={n_valid:5d}  MAE={mae:8.2f}  R2={r2:.3f}")
        models[target] = rf

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(TARGETS_PATH, "w") as f:
        json.dump(TARGET_COLS, f, indent=2)

    print(f"\n  Saved -> {MODEL_PATH}")
    return {"n_total": len(df), "metrics": metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def load() -> None:
    """Load the saved per-target model dict + column lists into module globals."""
    global _model, _feature_names, _target_names
    for p in (MODEL_PATH, FEATURES_PATH, TARGETS_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Personalized nutrition model artifact missing: {p}\n"
                f"Run: python -m app.services.personalized_nutrition"
            )
    _model = joblib.load(MODEL_PATH)   # dict {target_name: RandomForestRegressor}
    with open(FEATURES_PATH) as f:
        _feature_names = json.load(f)
    with open(TARGETS_PATH) as f:
        _target_names = json.load(f)
    print(f"[ok] Personalized nutrition model loaded "
          f"({len(_feature_names)} features, {len(_target_names)} per-target models)")


def predict_user_targets(user_profile: dict) -> dict[str, float]:
    """
    Predict the 8 daily nutrient targets for a single user profile.
    Each target has its own model — macros trained on meals-only rows,
    sodium/sugar/cholesterol trained on all rows.
    """
    if _model is None:
        raise RuntimeError(
            "Personalized model not loaded. Call personalized_nutrition.load() first."
        )

    row = _build_feature_row(user_profile)
    X = pd.DataFrame([row])[_feature_names].to_numpy()
    return {
        name: round(float(_model[name].predict(X)[0]), 1)
        for name in _target_names
    }


def enrich_with_targets(food_nutrients: dict, user_profile: dict) -> dict:
    """
    Add personalized data to any food check (manual / image / OCR).

    Computes:
      - user_targets:         predicted 8 daily nutrient limits
      - budget_impact:        food_value / daily_target per nutrient (fraction)
      - personalized_reasons: extra reason strings tailored to disease budgets

    Used by the routers AFTER hybrid_verdict() runs the safety logic
    (allergies, BMI, ML). This adds the "% of YOUR daily target" context
    on top, without replacing the rule engine.
    """
    user_targets = predict_user_targets(user_profile)

    nutrient_keys = ("calories", "protein", "carbs", "fat",
                     "fiber", "sugar", "sodium", "cholesterol")

    budget_impact: dict[str, float] = {}
    nutrients_for_reasoning: dict[str, float] = {}
    for key in nutrient_keys:
        try:
            value = float(food_nutrients.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0.0
        nutrients_for_reasoning[key] = value
        limit = user_targets.get(f"target_{key}", 0)
        budget_impact[key] = round(value / limit, 3) if limit > 0 else 0.0

    flags_row = _build_feature_row(user_profile)
    flag_dict = {k: flags_row[k] for k in
                 ("is_diabetes", "is_hypertension", "is_heart",
                  "is_weight_gain", "is_kidney")}

    from app.services.medical_rules import generate_reasoning
    reasoning = generate_reasoning(nutrients_for_reasoning, user_targets, flag_dict)

    return {
        "user_targets":          user_targets,
        "budget_impact":         budget_impact,
        "personalized_warnings": reasoning["warnings"],
        "personalized_reasons":  reasoning["reasons"],
        "personalized_score":    reasoning["score"],
    }


def check_personalized_verdict(food_item_name: str, user_profile: dict) -> dict:
    """
    Look up `food_item_name` in food_db_final_.csv, predict the user's daily
    targets, and report what fraction of each target this single food consumes.
    Includes a verdict + reasoning array tailored to the user's diseases.

    Returns:
      {
        "food_item":       str,
        "food_nutrients":  dict,        # nutrient profile from food DB
        "user_targets":    dict,        # predicted daily limits
        "budget_impact":   dict,        # {nutrient: pct_of_daily_limit}
        "verdict":         str,         # Safe | Caution | Avoid
        "score":           int,
        "reasons":         list[str],
      }
    """
    # Local import so this module can be imported standalone for training
    # without dragging in the FastAPI service stack.
    from app.services import food_lookup

    nutrients = food_lookup.lookup(food_item_name)
    if nutrients is None:
        return {
            "food_item":      food_item_name,
            "food_nutrients": None,
            "user_targets":   None,
            "budget_impact":  None,
            "verdict":        "Unknown",
            "score":          0,
            "reasons":        [f"Food '{food_item_name}' not found in database."],
        }

    user_targets = predict_user_targets(user_profile)

    # Build food_nutrients dict using the same nutrient keys used elsewhere
    food_nutrients = {
        "calories":    nutrients["calories"],
        "protein":     nutrients["protein"],
        "carbs":       nutrients["carbs"],
        "fat":         nutrients["fat"],
        "fiber":       nutrients["fiber"],
        "sugar":       nutrients["sugar"],
        "sodium":      nutrients["sodium"],
        "cholesterol": nutrients["cholesterol"],
    }

    # Budget impact: food_value / daily_target  (as fraction 0..1+)
    budget_impact: dict[str, float] = {}
    for nutrient, food_val in food_nutrients.items():
        limit = user_targets.get(f"target_{nutrient}", 0)
        budget_impact[nutrient] = round(float(food_val) / limit, 3) if limit > 0 else 0.0

    # Build flags dict the reasoning function expects
    profile_flags = _build_feature_row(user_profile)
    reasoning_profile = {
        k: profile_flags[k] for k in
        ("is_diabetes", "is_hypertension", "is_heart", "is_weight_gain", "is_kidney")
    }

    # Defer import to break a circular dependency at module load
    from app.services.medical_rules import generate_reasoning
    reasoning = generate_reasoning(food_nutrients, user_targets, reasoning_profile)

    return {
        "food_item":      nutrients["food_item"],
        "food_nutrients": food_nutrients,
        "user_targets":   user_targets,
        "budget_impact":  budget_impact,
        "verdict":        reasoning["verdict"],
        "score":          reasoning["score"],
        "reasons":        reasoning["reasons"],
    }


# ─── CLI entrypoint ──────────────────────────────────────────────────────────
# Usage: cd backend && python -m app.services.personalized_nutrition
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Training personalized nutrition model...\n")
    result = train_and_save()
    print(f"\nDone. Trained on {result['n_total']} rows.")

    print("\nQuick inference test:")
    load()
    sample_profile = {
        "age": 55, "gender": "male",
        "height": 172, "weight": 92, "bmi": 31.1,
        "activity_level": "Sedentary",
        "diseases": ["Diabetes", "Hypertension"],
    }
    targets = predict_user_targets(sample_profile)
    print(f"  Predicted daily targets for sample profile:")
    for k, v in targets.items():
        print(f"    {k:22s}  {v}")
