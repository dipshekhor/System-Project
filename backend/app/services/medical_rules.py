"""
medical_rules.py
================
Day 3–4 · Medical Rules Engine
--------------------------------
This is the HEART of the app. It answers one question:
  "Given a food's nutrients + a user's medical profile,
   should they eat this food?"

How it works:
  1. ALLERGY CHECK  → immediate hard block if any allergen found in ingredients
  2. DISEASE RULES  → deduct from score based on nutrient violations per disease
  3. POSITIVE BONUS → add points for fiber, protein, low sodium (healthy signals)
  4. VERDICT        → safe (≥70), caution (40–69), avoid (<40)

This is RULE-BASED (no ML here). The ML model in Phase 2 runs on top of this
and blends its prediction for better accuracy. Rules always override ML for
critical safety violations like allergies.

Data source: thresholds derived from Personalized_Diet_Recommendations.csv
  (75th percentile of recommended macros per disease = generous upper limit)
  Combined with clinical guidelines (AHA, WHO, NKF).
"""

# ─── Allergy keyword map ──────────────────────────────────────────────────────
# Maps each allergy name (as it appears in the dataset) to a list of ingredient
# keywords to scan for. We check these against raw OCR text or ingredient strings.
ALLERGY_KEYWORDS: dict[str, list[str]] = {
    "Nut Allergy": [
        "almond", "walnut", "cashew", "peanut", "pecan",
        "pistachio", "hazelnut", "macadamia", "chestnut",
        "nut", "nuts", "groundnut",
    ],
    "Gluten Intolerance": [
        "wheat", "gluten", "barley", "rye", "spelt",
        "semolina", "flour", "bread", "pasta", "oat",
        "malt", "bulgur", "couscous",
    ],
    "Lactose Intolerance": [
        "milk", "dairy", "cheese", "butter", "cream",
        "yogurt", "yoghurt", "lactose", "whey", "casein",
        "ghee", "curd", "skimmed milk", "condensed milk",
    ],
}


# ─── Disease-specific nutrient rules ─────────────────────────────────────────
# Each key maps to a dict of (nutrient → limit) pairs with penalty scores.
#
# Threshold sources:
#   Hypertension  → AHA: <1500mg sodium strict, <2300mg standard
#   Diabetes      → WHO: <25g added sugar/day; ADA: 130–225g carbs/day
#   Heart Disease → AHA: <200mg dietary cholesterol; <65g fat/day
#   Obesity       → General: caloric restriction, low fat, low sugar
#   Kidney Disease→ NKF: <2g sodium, controlled protein (0.6–0.8g/kg)
#
# penalty_score: how many points to deduct from the 100-point score
# per violation. Multiple violations stack.

DISEASE_RULES: dict[str, list[dict]] = {
    "Diabetes": [
        # High sugar spikes blood glucose directly
        {
            "nutrient": "sugar",
            "max": 10,                    # per-serving limit (g)
            "penalty": 30,
            "message": "High sugar content — raises blood glucose for Diabetes",
        },
        # High carbs also spike glucose (glycemic load)
        {
            "nutrient": "carbs",
            "max": 60,                    # per-serving limit (g)
            "penalty": 20,
            "message": "High carbohydrates — impacts blood sugar for Diabetes",
        },
    ],
    "Hypertension": [
        # Sodium directly raises blood pressure
        {
            "nutrient": "sodium",
            "max": 600,                   # per-serving limit (mg) — ~40% of 1500mg daily
            "penalty": 35,
            "message": "High sodium — dangerous for Hypertension (raises BP)",
        },
        # Saturated/total fat indirectly raises BP via arterial stiffness
        {
            "nutrient": "fat",
            "max": 20,
            "penalty": 15,
            "message": "High fat content — not ideal for Hypertension",
        },
    ],
    "Heart Disease": [
        # Dietary cholesterol directly impacts LDL
        {
            "nutrient": "cholesterol",
            "max": 75,                    # per-serving limit (mg) — ~37% of 200mg daily
            "penalty": 30,
            "message": "High cholesterol — risky for Heart Disease",
        },
        # Fat (especially saturated) builds arterial plaque
        {
            "nutrient": "fat",
            "max": 20,
            "penalty": 25,
            "message": "High fat — increases cardiovascular risk",
        },
        # Sodium raises blood pressure, stressing the heart
        {
            "nutrient": "sodium",
            "max": 800,
            "penalty": 20,
            "message": "High sodium — strains the heart",
        },
    ],
    "Obesity": [
        # Caloric restriction is the primary intervention
        {
            "nutrient": "calories",
            "max": 500,                   # per-serving limit (kcal)
            "penalty": 25,
            "message": "High calorie content — hinders weight management",
        },
        # High fat = calorie-dense (9 kcal/g vs 4 kcal/g for protein/carbs)
        {
            "nutrient": "fat",
            "max": 20,
            "penalty": 20,
            "message": "High fat content — calorie-dense for Obesity",
        },
        # Added sugars → fat storage via insulin
        {
            "nutrient": "sugar",
            "max": 10,
            "penalty": 15,
            "message": "High sugar — promotes fat storage for Obesity",
        },
    ],
    "Kidney Disease": [
        # Damaged kidneys cannot filter excess sodium
        {
            "nutrient": "sodium",
            "max": 500,                   # stricter than hypertension
            "penalty": 35,
            "message": "High sodium — kidneys cannot filter excess sodium",
        },
        # High protein increases urea/creatinine load on kidneys
        {
            "nutrient": "protein",
            "max": 20,                    # per-serving limit (g)
            "penalty": 30,
            "message": "High protein — increases kidney filtration burden",
        },
    ],
}


# ─── Positive nutrient bonuses ────────────────────────────────────────────────
# Foods with these properties get a score boost and a positive reason shown to user.
# Helps distinguish "no violations but still mediocre" from "genuinely healthy".
POSITIVE_RULES: list[dict] = [
    {
        "nutrient": "fiber",
        "min": 5,           # ≥5g fiber/serving = good source
        "bonus": 10,
        "message": "Good source of fiber (aids digestion and glucose control)",
    },
    {
        "nutrient": "protein",
        "min": 20,          # ≥20g protein/serving = high protein
        "bonus": 8,
        "message": "High protein content (supports muscle and satiety)",
    },
    {
        "nutrient": "sodium",
        "max": 140,         # ≤140mg = FDA 'low sodium'
        "bonus": 7,
        "message": "Low sodium (heart-friendly)",
    },
    {
        "nutrient": "sugar",
        "max": 5,           # ≤5g = low sugar
        "bonus": 5,
        "message": "Low sugar content",
    },
]


def _check_allergy(ingredients_text: str, allergies: list[str]) -> dict | None:
    """
    Scan raw ingredient text for allergens.
    Returns a block result immediately if any allergen keyword is found.
    This is a HARD BLOCK — allergy violations cannot be overridden by ML score.

    Args:
        ingredients_text: raw OCR text or ingredient string (lowercased)
        allergies: list of allergy names from user profile
                   e.g. ["Nut Allergy", "Gluten Intolerance"]

    Returns:
        dict with verdict='avoid' if allergen found, else None
    """
    if not ingredients_text or not allergies:
        return None

    text = ingredients_text.lower()
    for allergy in allergies:
        keywords = ALLERGY_KEYWORDS.get(allergy, [])
        for keyword in keywords:
            if keyword in text:
                return {
                    "verdict": "avoid",
                    "score": 0,
                    "warnings": [
                        f"⚠ ALLERGY ALERT: Contains '{keyword}' — "
                        f"triggers {allergy}. Do NOT consume."
                    ],
                    "reasons": [],
                    "allergy_block": True,   # flag so hybrid_verdict skips ML
                }
    return None


def check_verdict(food_nutrients: dict, user_profile: dict) -> dict:
    """
    Main verdict function. Combines allergy check + disease rules + positive bonuses.

    Args:
        food_nutrients: dict with keys:
            calories, protein, carbs, fat, fiber, sugar,
            sodium, cholesterol, ingredients_text (optional)
        user_profile: dict with keys:
            diseases  (list of str) e.g. ["Diabetes", "Hypertension"]
            allergies (list of str) e.g. ["Nut Allergy"]

    Returns:
        {
          "verdict":  "safe" | "caution" | "avoid",
          "score":    int (0–100),
          "warnings": list[str],   # reasons the food is bad
          "reasons":  list[str],   # reasons the food is good
          "allergy_block": bool    # True = skip ML, hard block
        }

    Scoring:
        Start at 100.
        Each disease rule violation deducts its penalty.
        Each positive rule adds its bonus (capped so total stays ≤ 100).
        Final: ≥70 = safe, 40–69 = caution, <40 = avoid
    """
    warnings: list[str] = []
    reasons:  list[str] = []
    score: int = 100

    diseases:  list[str] = user_profile.get("diseases",  [])
    allergies: list[str] = user_profile.get("allergies", [])
    ingredients_text: str = food_nutrients.get("ingredients_text", "")

    # ── Step 1: Hard allergy block ────────────────────────────────────────────
    allergy_result = _check_allergy(ingredients_text, allergies)
    if allergy_result:
        return allergy_result  # immediate return, no further checks needed

    # ── Step 2: Disease rule violations ──────────────────────────────────────
    for disease in diseases:
        rules = DISEASE_RULES.get(disease, [])
        for rule in rules:
            nutrient = rule["nutrient"]
            value    = float(food_nutrients.get(nutrient, 0))

            # Check max limit (penalty if exceeded)
            if "max" in rule and value > rule["max"]:
                warnings.append(
                    f"{rule['message']} "
                    f"(this food: {value:.1f}, limit: {rule['max']})"
                )
                score -= rule["penalty"]

            # Check min requirement (penalty if below)
            # (Currently only used for positive rules, but structure supports it)
            if "min" in rule and value < rule["min"]:
                warnings.append(
                    f"Insufficient {nutrient} for {disease} "
                    f"(this food: {value:.1f}, need ≥{rule['min']})"
                )
                score -= rule["penalty"]

    # ── Step 3: Positive nutrient bonuses ────────────────────────────────────
    for rule in POSITIVE_RULES:
        nutrient = rule["nutrient"]
        value    = float(food_nutrients.get(nutrient, 0))

        # Bonus for being below a good threshold
        if "max" in rule and value <= rule["max"]:
            reasons.append(rule["message"])
            score += rule["bonus"]

        # Bonus for being above a good minimum
        elif "min" in rule and value >= rule["min"]:
            reasons.append(rule["message"])
            score += rule["bonus"]

    # ── Step 4: Clamp score and assign verdict ────────────────────────────────
    score = max(0, min(100, score))   # keep in [0, 100]

    if score >= 70:
        verdict = "safe"
    elif score >= 40:
        verdict = "caution"
    else:
        verdict = "avoid"

    return {
        "verdict":       verdict,
        "score":         score,
        "warnings":      warnings,
        "reasons":       reasons,
        "allergy_block": False,
    }


def hybrid_verdict(
    food_nutrients: dict,
    user_profile: dict,
    ml_predict_fn=None,          # optional: callable(nutrients, disease_flags) -> str
) -> dict:
    """
    Combines rule-based check_verdict() with optional ML model prediction.

    Why hybrid?
      - Rules alone are rigid (miss subtle patterns in data)
      - ML alone may hallucinate (no hard safety guarantees)
      - Together: rules enforce safety, ML adds nuance

    Blend weights: 60% rule score + 40% ML score
    Allergy violations always override ML (score=0, verdict=avoid).

    Args:
        food_nutrients:  same as check_verdict
        user_profile:    same as check_verdict
        ml_predict_fn:   function from ml_model.py loaded at startup
                         Pass None to skip ML (e.g. during testing)

    Returns:
        Same shape as check_verdict, with optional 'ml_prediction' key added.
    """
    # Get rule-based result first
    rule_result = check_verdict(food_nutrients, user_profile)

    # Hard allergy block — never let ML override this
    if rule_result.get("allergy_block"):
        return rule_result

    # If no ML function provided, return rule result as-is
    if ml_predict_fn is None:
        return rule_result

    # Build disease flags for ML feature vector
    # e.g. ["Diabetes", "Hypertension"] → {"diabetes": 1, "hypertension": 1}
    disease_flags = {
        d.lower().replace(" ", "_"): 1
        for d in user_profile.get("diseases", [])
    }

    # Get ML prediction
    ml_pred  = ml_predict_fn(food_nutrients, disease_flags)
    ml_score = {"safe": 100, "caution": 60, "avoid": 20}.get(ml_pred, 60)

    # Blend: rule score has higher weight (safety-critical)
    blended = round(0.6 * rule_result["score"] + 0.4 * ml_score)
    blended = max(0, min(100, blended))

    if blended >= 70:
        final_verdict = "safe"
    elif blended >= 40:
        final_verdict = "caution"
    else:
        final_verdict = "avoid"

    rule_result["verdict"]       = final_verdict
    rule_result["score"]         = blended
    rule_result["ml_prediction"] = ml_pred  # expose for debugging / UI

    return rule_result


# ─── Quick self-test (run this file directly to verify rules work) ────────────
if __name__ == "__main__":
    print("=" * 55)
    print("medical_rules.py — self-test")
    print("=" * 55)

    # Test 1: Hypertension + high sodium → should be AVOID
    result1 = check_verdict(
        food_nutrients={"calories": 350, "fat": 12, "sodium": 1800,
                        "sugar": 5, "cholesterol": 60, "carbs": 40,
                        "protein": 18, "fiber": 2, "ingredients_text": ""},
        user_profile={"diseases": ["Hypertension"], "allergies": []},
    )
    print(f"\nTest 1 — Hypertension + 1800mg sodium")
    print(f"  Verdict : {result1['verdict']}  (expected: avoid or caution)")
    print(f"  Score   : {result1['score']}")
    print(f"  Warnings: {result1['warnings']}")

    # Test 2: Nut allergy + almond in ingredients → hard block
    result2 = check_verdict(
        food_nutrients={"calories": 200, "fat": 8, "sodium": 100,
                        "sugar": 4, "cholesterol": 0, "carbs": 25,
                        "protein": 6, "fiber": 2,
                        "ingredients_text": "oat flour, sugar, almonds, salt"},
        user_profile={"diseases": [], "allergies": ["Nut Allergy"]},
    )
    print(f"\nTest 2 — Nut allergy + almonds in ingredients")
    print(f"  Verdict : {result2['verdict']}  (expected: avoid)")
    print(f"  Score   : {result2['score']}   (expected: 0)")
    print(f"  Warnings: {result2['warnings']}")

    # Test 3: Healthy food, no diseases → should be SAFE
    result3 = check_verdict(
        food_nutrients={"calories": 180, "fat": 6, "sodium": 80,
                        "sugar": 3, "cholesterol": 40, "carbs": 15,
                        "protein": 22, "fiber": 6, "ingredients_text": ""},
        user_profile={"diseases": [], "allergies": []},
    )
    print(f"\nTest 3 — Healthy food, no conditions")
    print(f"  Verdict : {result3['verdict']}  (expected: safe)")
    print(f"  Score   : {result3['score']}")
    print(f"  Reasons : {result3['reasons']}")

    # Test 4: Diabetes + high sugar + high carbs
    result4 = check_verdict(
        food_nutrients={"calories": 420, "fat": 5, "sodium": 120,
                        "sugar": 35, "cholesterol": 0, "carbs": 90,
                        "protein": 4, "fiber": 1, "ingredients_text": ""},
        user_profile={"diseases": ["Diabetes"], "allergies": []},
    )
    print(f"\nTest 4 — Diabetes + 35g sugar + 90g carbs")
    print(f"  Verdict : {result4['verdict']}  (expected: avoid)")
    print(f"  Score   : {result4['score']}")
    print(f"  Warnings: {result4['warnings']}")

    print("\n✓ All tests complete.")
