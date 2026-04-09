"""
ml_model.py
===========
Day 9–11 · ML Model Service Wrapper
-------------------------------------
This module does TWO things:

  1. TRAINING  → train_and_save()
     Reads merged_training_data.csv (produced by 02_data_preprocessing.ipynb),
     trains a RandomForestClassifier, and saves the model + encoder + feature list
     to backend/app/ml_models/ as .pkl files.
     Run this ONCE before starting the FastAPI server.

  2. INFERENCE → load() + predict()
     Loads the saved .pkl files into memory once at FastAPI startup (lifespan).
     predict() is called on every /check-food, /analyze-ocr, /analyze-image request.

Why RandomForest?
  - Handles class imbalance well with class_weight='balanced'
  - No feature scaling needed (unlike SVM/LogReg)
  - Gives feature importances for debugging
  - Fast inference (<<1ms per prediction)
  - Works well on our 5310-row dataset without overfitting

The model predicts: 'safe' | 'caution' | 'avoid'
Features: [calories, protein, carbs, fat, fiber, sugar, sodium, cholesterol,
           has_obesity, has_hypertension, has_heart_disease, has_kidney_disease, has_diabetes]

This model is the ML half of hybrid_verdict() in medical_rules.py.
Rules handle hard violations (allergy, extreme sodium). ML adds nuance.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
# All paths relative to this file's location → works regardless of cwd
_THIS_DIR    = Path(__file__).parent
_DATA_DIR    = _THIS_DIR.parent / "data"
_MODELS_DIR  = _THIS_DIR.parent / "ml_models"

MODEL_PATH    = _MODELS_DIR / "food_suitability_model.pkl"
ENCODER_PATH  = _MODELS_DIR / "label_encoder.pkl"
FEATURES_PATH = _MODELS_DIR / "feature_names.json"

# ─── Module-level singletons (loaded once at startup) ─────────────────────────
_model         = None   # RandomForestClassifier
_label_encoder = None   # LabelEncoder (maps 0/1/2 → avoid/caution/safe)
_feature_names = None   # list[str] — column order the model was trained on


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: Training
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save(data_path: Path | str | None = None) -> dict:
    """
    Train the food suitability classifier and save all artifacts.

    This function:
      1. Generates training data by combining food_db_final_.csv ×
         disease profile combinations (590 foods × 9 profiles = 5,310 rows)
      2. Labels each row using the medical_rules engine (programmatic labeling)
      3. Trains RandomForestClassifier with class_weight='balanced' to handle
         the safe>>caution>>avoid class imbalance
      4. Evaluates with 5-fold cross-validation
      5. Saves model.pkl, encoder.pkl, feature_names.json

    Args:
        data_path: path to food_db_final_.csv. Defaults to backend/app/data/

    Returns:
        dict with cv_accuracy, cv_std, class_report
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report
    # Import medical rules to generate labels (avoids duplicate logic)
    from app.services.medical_rules import check_verdict

    print("=" * 55)
    print("Training food suitability ML model")
    print("=" * 55)

    # ── Step 1: Load food database ────────────────────────────────────────────
    food_csv = Path(data_path) if data_path else (_DATA_DIR / "food_db_final_.csv")
    food_df  = pd.read_csv(food_csv)

    # Normalize column names to snake_case (same logic as food_lookup.load)
    food_df.columns = (
        food_df.columns
        .str.strip().str.lower()
        .str.replace(r"[\s]+", "_", regex=True)
        .str.replace(r"[()\/]", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    # Drop empty trailing column
    food_df = food_df.drop(
        columns=[c for c in food_df.columns if c.startswith("unnamed")],
        errors="ignore"
    )
    print(f"  Food DB loaded: {len(food_df)} items")

    # ── Step 2: Define disease profiles for training ──────────────────────────
    # We generate one training row per food × per disease profile combination.
    # This gives the model examples of EVERY food in EVERY disease context.
    # 590 foods × 9 profiles = 5,310 training rows
    PROFILES = [
        # (list_of_diseases,             flags: [obesity, hypertension, heart, kidney, diabetes])
        ([],                             [0, 0, 0, 0, 0]),   # healthy baseline
        (["Diabetes"],                   [0, 0, 0, 0, 1]),
        (["Hypertension"],               [0, 1, 0, 0, 0]),
        (["Heart Disease"],              [0, 0, 1, 0, 0]),
        (["Obesity"],                    [1, 0, 0, 0, 0]),
        (["Kidney Disease"],             [0, 0, 0, 1, 0]),
        (["Diabetes", "Hypertension"],   [0, 1, 0, 0, 1]),
        (["Heart Disease", "Obesity"],   [1, 0, 1, 0, 0]),
        (["Hypertension", "Heart Disease"], [0, 1, 1, 0, 0]),
    ]

    # ── Step 3: Generate labeled rows ────────────────────────────────────────
    # For each food × each disease profile:
    #   - Build nutrients dict from food_db row
    #   - Build profile dict from disease list
    #   - Call check_verdict() → gets 'safe'/'caution'/'avoid' label
    rows = []
    for _, frow in food_df.iterrows():
        # Extract nutrient values from food_db row
        nutrients = {
            "calories":    float(frow.get("calories_kcal", 0) or 0),
            "protein":     float(frow.get("protein_g", 0) or 0),
            "carbs":       float(frow.get("carbohydrates_g", 0) or 0),
            "fat":         float(frow.get("fat_g", 0) or 0),
            "fiber":       float(frow.get("fiber_g", 0) or 0),
            "sugar":       float(frow.get("sugars_g", 0) or 0),
            "sodium":      float(frow.get("sodium_mg", 0) or 0),
            "cholesterol": float(frow.get("cholesterol_mg", 0) or 0),
            "ingredients_text": "",  # no ingredient text for food_db rows
        }
        for diseases, flags in PROFILES:
            # Generate label via rule engine
            profile = {"diseases": diseases, "allergies": []}
            result  = check_verdict(nutrients, profile)
            label   = result["verdict"]   # 'safe' | 'caution' | 'avoid'

            rows.append({
                # ── Nutrient features ───────────────────────────
                "calories":    nutrients["calories"],
                "protein":     nutrients["protein"],
                "carbs":       nutrients["carbs"],
                "fat":         nutrients["fat"],
                "fiber":       nutrients["fiber"],
                "sugar":       nutrients["sugar"],
                "sodium":      nutrients["sodium"],
                "cholesterol": nutrients["cholesterol"],
                # ── Disease binary flags ─────────────────────────
                "has_obesity":        flags[0],
                "has_hypertension":   flags[1],
                "has_heart_disease":  flags[2],
                "has_kidney_disease": flags[3],
                "has_diabetes":       flags[4],
                # ── Label ────────────────────────────────────────
                "label": label,
            })

    df = pd.DataFrame(rows)
    print(f"\n  Training data shape: {df.shape}")
    print(f"  Label distribution:")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {label:8s}: {count:4d} ({pct:.1f}%)")

    # ── Step 4: Build feature matrix ──────────────────────────────────────────
    FEATURE_COLS = [
        "calories", "protein", "carbs", "fat", "fiber", "sugar",
        "sodium", "cholesterol",
        "has_obesity", "has_hypertension", "has_heart_disease",
        "has_kidney_disease", "has_diabetes",
    ]

    X = df[FEATURE_COLS].values
    # Encode labels: 'avoid'→0, 'caution'→1, 'safe'→2 (alphabetical by default)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    print(f"\n  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Step 5: Train/test split ──────────────────────────────────────────────
    # stratify=y ensures all 3 classes appear in both train and test sets.
    # Critical with imbalanced data — without stratify, 'avoid' rows may end up
    # only in train or only in test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Step 6: Train model ───────────────────────────────────────────────────
    # class_weight='balanced': automatically weights minority classes higher
    # so the model doesn't just predict 'safe' for everything.
    # n_estimators=200: 200 trees (more stable than 100, not much slower)
    # max_depth=12: limits tree depth to prevent overfitting on small minority classes
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,       # each leaf needs ≥2 samples (reduces overfit)
        class_weight="balanced",  # handles safe>>caution>>avoid imbalance
        random_state=42,
        n_jobs=-1,                # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("\n  Model trained.")

    # ── Step 7: Cross-validation ──────────────────────────────────────────────
    # StratifiedKFold preserves class ratios in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Step 8: Test set evaluation ───────────────────────────────────────────
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    print(f"\n  Classification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Step 9: Feature importances ───────────────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("  Feature importances (top 8):")
    for feat, imp in importances.sort_values(ascending=False).head(8).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:20s} {imp:.4f} {bar}")

    # ── Step 10: Save artifacts ───────────────────────────────────────────────
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le,    ENCODER_PATH)

    # Save feature names as JSON (human-readable, not pickle)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"\n  Saved:")
    print(f"    {MODEL_PATH}")
    print(f"    {ENCODER_PATH}")
    print(f"    {FEATURES_PATH}")

    return {
        "cv_accuracy":   float(cv_scores.mean()),
        "cv_std":        float(cv_scores.std()),
        "class_report":  report,
        "n_training":    len(df),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: Inference (used by FastAPI at runtime)
# ──────────────────────────────────────────────────────────────────────────────

def load() -> None:
    """
    Load saved model artifacts into module-level globals.
    Called ONCE in FastAPI lifespan (main.py) at server startup.
    After this, predict() can be called on every request without re-loading.

    Raises:
        FileNotFoundError: if .pkl files don't exist yet (run train_and_save first)
    """
    global _model, _label_encoder, _feature_names

    for path in [MODEL_PATH, ENCODER_PATH, FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"ML model file not found: {path}\n"
                f"Run: python -m app.services.ml_model\n"
                f"  or call train_and_save() to generate it first."
            )

    _model         = joblib.load(MODEL_PATH)
    _label_encoder = joblib.load(ENCODER_PATH)

    with open(FEATURES_PATH) as f:
        _feature_names = json.load(f)

    print(f"✓ ML model loaded ({len(_feature_names)} features)")


def predict(nutrients: dict, disease_flags: dict) -> str:
    """
    Predict food suitability for a given nutrient profile + disease flags.
    Called by hybrid_verdict() in medical_rules.py.

    Args:
        nutrients:     dict with nutrient keys (same as returned by food_lookup.lookup)
                       e.g. {'calories': 350, 'sodium': 800, 'fat': 20, ...}
        disease_flags: dict of binary disease indicators
                       e.g. {'has_diabetes': 1, 'has_hypertension': 1}
                       Keys must match what was used during training.

    Returns:
        'safe' | 'caution' | 'avoid'

    Raises:
        RuntimeError: if load() hasn't been called yet
    """
    if _model is None:
        raise RuntimeError(
            "ML model not loaded. Call ml_model.load() at startup."
        )

    # Build feature row in EXACT column order used during training.
    # If a feature is missing from inputs, default to 0.
    # Using dict + reindex ensures column order matches training even if
    # the caller passes extra or missing keys.
    row = {}
    for feature in _feature_names:
        # Check nutrients dict first, then disease_flags dict
        if feature in nutrients:
            row[feature] = float(nutrients[feature] or 0)
        elif feature in disease_flags:
            row[feature] = float(disease_flags[feature] or 0)
        else:
            row[feature] = 0.0

    # Build single-row DataFrame — RandomForest expects 2D input
    X = pd.DataFrame([row])[_feature_names].to_numpy()

    # Predict class index → decode to label string
    pred_idx  = _model.predict(X)[0]
    pred_label = _label_encoder.inverse_transform([pred_idx])[0]

    return pred_label  # 'safe' | 'caution' | 'avoid'


def predict_proba(nutrients: dict, disease_flags: dict) -> dict:
    """
    Return prediction probabilities for all 3 classes.
    Useful for showing confidence on the frontend
    (e.g. "87% safe" vs just "safe").

    Args:
        Same as predict()

    Returns:
        dict: {'avoid': 0.05, 'caution': 0.08, 'safe': 0.87}
    """
    if _model is None:
        raise RuntimeError("ML model not loaded. Call ml_model.load() first.")

    row = {}
    for feature in _feature_names:
        if feature in nutrients:
            row[feature] = float(nutrients[feature] or 0)
        elif feature in disease_flags:
            row[feature] = float(disease_flags[feature] or 0)
        else:
            row[feature] = 0.0

    X      = pd.DataFrame([row])[_feature_names]
    proba  = _model.predict_proba(X)[0]   # array of probabilities per class
    classes = _label_encoder.classes_     # ['avoid', 'caution', 'safe'] (alphabetical)

    return {cls: round(float(prob), 4) for cls, prob in zip(classes, proba)}


# ─── Entry point: run this file to train and save the model ──────────────────
# Usage: cd backend && python -m app.services.ml_model
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Running ml_model.py directly → training model...\n")
    results = train_and_save()
    print(f"\nDone. CV Accuracy: {results['cv_accuracy']:.3f} ± {results['cv_std']:.3f}")
    print(f"Training rows used: {results['n_training']}")

    # Quick inference test
    print("\nRunning inference test after training...")
    load()

    # Test 1: Healthy food, no disease → should be safe
    r1 = predict(
        nutrients={"calories": 180, "protein": 22, "carbs": 15,
                   "fat": 6, "fiber": 6, "sugar": 3, "sodium": 80, "cholesterol": 40},
        disease_flags={"has_diabetes": 0, "has_hypertension": 0,
                       "has_heart_disease": 0, "has_kidney_disease": 0, "has_obesity": 0},
    )
    print(f"  Test 1 (healthy food, no disease): {r1}  (expected: safe)")

    # Test 2: High sodium food, hypertension patient → should be avoid
    r2 = predict(
        nutrients={"calories": 350, "protein": 12, "carbs": 40,
                   "fat": 15, "fiber": 1, "sugar": 5, "sodium": 1500, "cholesterol": 60},
        disease_flags={"has_diabetes": 0, "has_hypertension": 1,
                       "has_heart_disease": 0, "has_kidney_disease": 0, "has_obesity": 0},
    )
    print(f"  Test 2 (high sodium + hypertension): {r2}  (expected: avoid)")

    # Test 3: Probabilities
    p3 = predict_proba(
        nutrients={"calories": 200, "protein": 18, "carbs": 25,
                   "fat": 8, "fiber": 4, "sugar": 6, "sodium": 400, "cholesterol": 50},
        disease_flags={"has_diabetes": 1, "has_hypertension": 0,
                       "has_heart_disease": 0, "has_kidney_disease": 0, "has_obesity": 0},
    )
    print(f"  Test 3 probabilities: {p3}")
    print("\n✓ ml_model.py ready.")
