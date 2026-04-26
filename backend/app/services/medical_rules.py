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
  2. BMI CHECK      → auto-inject Obesity if BMI ≥ 30; flag overweight if BMI ≥ 25
  3. AGE/GENDER     → scale calorie + sodium limits to estimated TDEE
  4. DISEASE RULES  → deduct from score based on nutrient violations per disease
  5. POSITIVE BONUS → add points for fiber, protein, low sodium (healthy signals)
  6. VERDICT        → safe (≥70), caution (40–69), avoid (<40)

This is RULE-BASED (no ML here). The ML model in Phase 2 runs on top of this
and blends its prediction for better accuracy. Rules always override ML for
critical safety violations like allergies.

Data source: thresholds derived from Personalized_Diet_Recommendations.csv
  (75th percentile of recommended macros per disease = generous upper limit)
  Combined with clinical guidelines (AHA, WHO, NKF, Harris-Benedict).
"""

# ─── BMI helpers ─────────────────────────────────────────────────────────────

def compute_bmi(height_cm: float, weight_kg: float) -> float | None:
    """Return BMI or None if inputs are missing/invalid."""
    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h > 0 and w > 0:
            return round(w / ((h / 100) ** 2), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def _bmi_category(bmi: float | None) -> str:
    """Return 'obese' | 'overweight' | 'normal'."""
    if bmi is None:
        return "normal"
    if bmi >= 30:
        return "obese"
    if bmi >= 25:
        return "overweight"
    return "normal"


# ─── Age/gender calorie + sodium scaling ─────────────────────────────────────
#
# Estimated daily calorie needs via simplified Harris-Benedict:
#   Men   : 88.4 + 13.4×kg + 4.8×cm − 5.7×age  (sedentary ×1.2)
#   Women : 447.6 + 9.2×kg + 3.1×cm − 4.3×age  (sedentary ×1.2)
#
# We then set the per-meal calorie limit = daily_need / 3 (3 meals/day).
# The Obesity rule default is 500 kcal; we replace it with this value.
#
# Sodium: AHA recommends stricter limits for people over 50 (cardiovascular
# risk rises sharply). Default Hypertension limit is 600 mg; we tighten it
# to 500 mg for age ≥ 50.

def _estimate_daily_calories(age: int, gender: str, height_cm: float, weight_kg: float) -> float | None:
    """Return estimated daily sedentary calorie need, or None if inputs missing."""
    try:
        a = float(age)
        h = float(height_cm)
        w = float(weight_kg)
        if a <= 0 or h <= 0 or w <= 0:
            return None
        g = str(gender).lower()
        if g in ("male", "m"):
            bmr = 88.4 + 13.4 * w + 4.8 * h - 5.7 * a
        else:
            bmr = 447.6 + 9.2 * w + 3.1 * h - 4.3 * a
        return round(bmr * 1.2, 0)   # sedentary activity factor
    except (TypeError, ValueError):
        return None


def _per_meal_calorie_limit(daily_calories: float | None) -> int:
    """One-third of daily need, clamped to [350, 700]."""
    if daily_calories is None:
        return 500   # fallback default
    return max(350, min(700, round(daily_calories / 3)))


def _sodium_limit_for_hypertension(age: int | None) -> int:
    """Stricter sodium cap for hypertension patients over 50."""
    try:
        if age is not None and int(age) >= 50:
            return 500
    except (TypeError, ValueError):
        pass
    return 600   # default


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
    Main verdict function. Combines allergy check + BMI injection +
    age/gender scaling + disease rules + positive bonuses.

    Args:
        food_nutrients: dict with keys:
            calories, protein, carbs, fat, fiber, sugar,
            sodium, cholesterol, ingredients_text (optional)
        user_profile: dict with keys:
            diseases   (list of str) e.g. ["Diabetes", "Hypertension"]
            allergies  (list of str) e.g. ["Nut Allergy"]
            age        (int,   optional) used to scale sodium limit
            gender     (str,   optional) "male"/"female" for calorie scaling
            height_cm  (float, optional) for BMI computation
            weight_kg  (float, optional) for BMI computation

    Returns:
        {
          "verdict":      "safe" | "caution" | "avoid",
          "score":        int (0–100),
          "warnings":     list[str],
          "reasons":      list[str],
          "allergy_block": bool,
          "bmi":          float | None,
          "bmi_note":     str | None,
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

    diseases:  list[str] = list(user_profile.get("diseases",  []))
    allergies: list[str] = user_profile.get("allergies", [])
    ingredients_text: str = food_nutrients.get("ingredients_text", "")

    age       = user_profile.get("age")
    gender    = user_profile.get("gender", "")
    height_cm = user_profile.get("height_cm")
    weight_kg = user_profile.get("weight_kg")

    # ── Step 1: Hard allergy block ────────────────────────────────────────────
    allergy_result = _check_allergy(ingredients_text, allergies)
    if allergy_result:
        return allergy_result

    # ── Step 2: BMI auto-injection ────────────────────────────────────────────
    bmi      = compute_bmi(height_cm, weight_kg)
    bmi_note = None
    bmi_cat  = _bmi_category(bmi)

    if bmi_cat == "obese" and "Obesity" not in diseases:
        diseases.append("Obesity")
        bmi_note = (
            f"Obesity rules applied automatically (BMI {bmi} ≥ 30). "
            "Add 'Obesity' to your profile to silence this notice."
        )
        warnings.append(f"⚠ BMI {bmi} indicates obesity — calorie and fat limits applied")

    elif bmi_cat == "overweight":
        # Softer penalty: deduct 10 points if calories or fat are high
        cal_val = float(food_nutrients.get("calories", 0))
        fat_val = float(food_nutrients.get("fat", 0))
        if cal_val > 600 or fat_val > 25:
            score -= 10
            warnings.append(
                f"Moderately high calorie/fat content — worth monitoring "
                f"(BMI {bmi}, overweight range)"
            )
        bmi_note = f"BMI {bmi} — overweight range. Lighter portions recommended."

    # ── Step 3: Age/gender-adjusted calorie + sodium limits ───────────────────
    daily_cal   = _estimate_daily_calories(age, gender, height_cm, weight_kg)
    meal_cal_limit  = _per_meal_calorie_limit(daily_cal)
    htn_sodium_limit = _sodium_limit_for_hypertension(age)

    # Build a patched copy of DISEASE_RULES with personalised limits
    personalized_rules: dict[str, list[dict]] = {}
    for disease, rules in DISEASE_RULES.items():
        patched = []
        for rule in rules:
            r = dict(rule)
            if disease == "Obesity" and r["nutrient"] == "calories":
                r = {**r, "max": meal_cal_limit}
            if disease == "Hypertension" and r["nutrient"] == "sodium":
                r = {**r, "max": htn_sodium_limit}
            patched.append(r)
        personalized_rules[disease] = patched

    # ── Step 4: Disease rule violations ──────────────────────────────────────
    # Score deductions are applied silently here; the personalized budget-impact
    # messages from generate_reasoning (via _enrich_result) carry the user-facing
    # explanation, so we don't double-report with the static limit strings.
    for disease in diseases:
        rules = personalized_rules.get(disease, [])
        for rule in rules:
            nutrient = rule["nutrient"]
            value    = float(food_nutrients.get(nutrient, 0))

            if "max" in rule and value > rule["max"]:
                score -= rule["penalty"]

            if "min" in rule and value < rule["min"]:
                score -= rule["penalty"]

    # ── Step 5: Positive nutrient bonuses ────────────────────────────────────
    for rule in POSITIVE_RULES:
        nutrient = rule["nutrient"]
        value    = float(food_nutrients.get(nutrient, 0))

        if "max" in rule and value <= rule["max"]:
            reasons.append(rule["message"])
            score += rule["bonus"]
        elif "min" in rule and value >= rule["min"]:
            reasons.append(rule["message"])
            score += rule["bonus"]

    # ── Step 6: Clamp score and assign verdict ────────────────────────────────
    score = max(0, min(100, score))

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
        "bmi":           bmi,
        "bmi_note":      bmi_note,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Personalized "budget impact" reasoning
# ──────────────────────────────────────────────────────────────────────────────
# Used by services/personalized_nutrition.check_personalized_verdict().
# Different from check_verdict() above:
#   check_verdict      → static thresholds (e.g. "sodium > 600mg = bad")
#   generate_reasoning → compares against the USER's predicted daily target
#                        (e.g. "this food consumes 42% of YOUR sodium budget")
#
# Disease sensitivities tighten the warning threshold for nutrients the
# condition is most affected by (sodium for hypertension, sugar for diabetes…).
_SENSITIVITY_MAP: dict[str, list[str]] = {
    "is_diabetes":     ["target_sugar", "target_carbs"],
    "is_hypertension": ["target_sodium", "target_cholesterol"],
    "is_heart":        ["target_fat", "target_cholesterol", "target_sodium"],
    "is_kidney":       ["target_sodium", "target_protein"],
    "is_weight_gain":  ["target_calories", "target_fat", "target_sugar"],
}


def generate_reasoning(
    food_nutrients: dict,
    user_targets:   dict,
    user_profile:   dict,
) -> dict:
    """
    Compare a food's nutrients against a user's PERSONAL daily targets.
    Returns warnings (over-budget violations) and positive reasons separately
    so the frontend can route them to the correct UI sections.

    Args:
        food_nutrients: {"calories": 320, "protein": 12, ...}
        user_targets:   {"target_calories": 2150, "target_protein": 108, ...}
        user_profile:   binary disease flags
                        {"is_diabetes": 1, "is_hypertension": 0, ...}

    Returns:
        {"verdict": "Safe"|"Caution"|"Avoid",
         "score":   int 0-100,
         "warnings": list[str],     # over-budget / disease-sensitive concerns
         "reasons":  list[str]}     # positive reasons only

    Scoring:
      Start at 100. Each high-impact nutrient deducts points proportional
      to its share of the daily target (capped per-nutrient so a single
      runaway value can't dominate). Disease-sensitive nutrients use a
      tighter 20% threshold and deduct an extra 15. Positive bonuses
      (high fiber, high protein) add a few points back.
    """
    warnings: list[str] = []
    reasons:  list[str] = []
    score: float = 100.0

    for nutrient, food_val in food_nutrients.items():
        target_key = f"target_{nutrient}"
        limit = user_targets.get(target_key, 0)
        if not limit or limit <= 0:
            continue

        try:
            food_val = float(food_val or 0)
        except (TypeError, ValueError):
            continue
        impact = food_val / limit
        # Cap impact display at 999% so a unit-mismatched outlier reads sanely
        impact_pct = min(impact, 9.99)

        # Generic score deduction: > 30% of daily limit in a single food item
        if impact > 0.30:
            # Cap deduction so a single runaway nutrient can't go past 25 points
            score -= min(25.0, impact * 20)

        # Stricter rule for disease-sensitive nutrients
        for disease, sensitive in _SENSITIVITY_MAP.items():
            if user_profile.get(disease) == 1 and target_key in sensitive and impact > 0.30:
                pretty = disease.replace("is_", "").replace("_", " ")
                warnings.append(
                    f"Critical for {pretty}: too much {nutrient} "
                    f"({impact_pct:.0%} of daily target)."
                )
                score -= 15

    # Positive reasons
    if float(food_nutrients.get("fiber", 0) or 0) > 5:
        reasons.append("Good source of fibre.")
        score += 5
    if float(food_nutrients.get("protein", 0) or 0) > 20:
        reasons.append("Good source of protein.")
        score += 3

    score = max(0, min(100, int(round(score))))
    if score >= 70:
        verdict = "Safe"
    elif score >= 40:
        verdict = "Caution"
    else:
        verdict = "Avoid"

    return {
        "verdict":  verdict,
        "score":    score,
        "warnings": warnings,
        "reasons":  reasons,
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

    # ── NEW: BMI + age/gender tests ───────────────────────────────────────────

    # Test 5: Obese user (BMI 33) who did NOT select Obesity disease
    # → should auto-inject Obesity rules and penalise high-cal food
    result5 = check_verdict(
        food_nutrients={"calories": 600, "fat": 25, "sodium": 200,
                        "sugar": 8, "cholesterol": 40, "carbs": 50,
                        "protein": 15, "fiber": 2, "ingredients_text": ""},
        user_profile={
            "diseases": [], "allergies": [],
            "height_cm": 170, "weight_kg": 95,   # BMI ≈ 32.9
            "age": 35, "gender": "male",
        },
    )
    bmi5 = compute_bmi(170, 95)
    print(f"\nTest 5 — Obese user (BMI {bmi5}), no diseases selected, high-cal food")
    print(f"  Verdict  : {result5['verdict']}  (expected: caution or avoid)")
    print(f"  Score    : {result5['score']}")
    print(f"  BMI      : {result5['bmi']}")
    print(f"  BMI note : {result5['bmi_note']}")
    print(f"  Warnings : {result5['warnings']}")

    # Test 6: Overweight user — lighter penalty only
    result6 = check_verdict(
        food_nutrients={"calories": 650, "fat": 28, "sodium": 200,
                        "sugar": 6, "cholesterol": 30, "carbs": 45,
                        "protein": 20, "fiber": 5, "ingredients_text": ""},
        user_profile={
            "diseases": [], "allergies": [],
            "height_cm": 175, "weight_kg": 85,   # BMI ≈ 27.8
            "age": 40, "gender": "female",
        },
    )
    bmi6 = compute_bmi(175, 85)
    print(f"\nTest 6 — Overweight user (BMI {bmi6}), high-cal food")
    print(f"  Verdict  : {result6['verdict']}  (expected: caution, soft penalty)")
    print(f"  Score    : {result6['score']}")
    print(f"  BMI note : {result6['bmi_note']}")
    print(f"  Warnings : {result6['warnings']}")

    # Test 7: Age/gender calorie scaling — older woman gets stricter calorie limit
    result7 = check_verdict(
        food_nutrients={"calories": 520, "fat": 15, "sodium": 300,
                        "sugar": 5, "cholesterol": 40, "carbs": 50,
                        "protein": 20, "fiber": 5, "ingredients_text": ""},
        user_profile={
            "diseases": ["Obesity"], "allergies": [],
            "height_cm": 158, "weight_kg": 72,
            "age": 60, "gender": "female",
        },
    )
    daily7 = _estimate_daily_calories(60, "female", 158, 72)
    meal_limit7 = _per_meal_calorie_limit(daily7)
    print(f"\nTest 7 — 60yr female, Obesity, 520 kcal food")
    print(f"  Daily estimate : {daily7} kcal  →  per-meal limit: {meal_limit7} kcal")
    print(f"  Verdict        : {result7['verdict']}")
    print(f"  Score          : {result7['score']}")
    print(f"  Warnings       : {result7['warnings']}")

    # Test 8: Hypertension + age 55 → stricter sodium limit (500mg not 600mg)
    result8 = check_verdict(
        food_nutrients={"calories": 300, "fat": 10, "sodium": 540,
                        "sugar": 4, "cholesterol": 30, "carbs": 30,
                        "protein": 18, "fiber": 3, "ingredients_text": ""},
        user_profile={
            "diseases": ["Hypertension"], "allergies": [],
            "height_cm": 172, "weight_kg": 78,
            "age": 55, "gender": "male",
        },
    )
    print(f"\nTest 8 — Hypertension, age 55, 540mg sodium food")
    print(f"  Sodium limit applied: {_sodium_limit_for_hypertension(55)}mg  (expected: 500mg)")
    print(f"  Verdict  : {result8['verdict']}  (expected: caution/avoid — 540 > 500)")
    print(f"  Score    : {result8['score']}")
    print(f"  Warnings : {result8['warnings']}")

    # Same food, younger user — 540mg should be under the 600mg default limit
    result8b = check_verdict(
        food_nutrients={"calories": 300, "fat": 10, "sodium": 540,
                        "sugar": 4, "cholesterol": 30, "carbs": 30,
                        "protein": 18, "fiber": 3, "ingredients_text": ""},
        user_profile={
            "diseases": ["Hypertension"], "allergies": [],
            "height_cm": 172, "weight_kg": 78,
            "age": 30, "gender": "male",
        },
    )
    print(f"\nTest 8b — Same food, age 30 (limit 600mg)")
    print(f"  Sodium limit applied: {_sodium_limit_for_hypertension(30)}mg  (expected: 600mg)")
    print(f"  Verdict  : {result8b['verdict']}  (expected: safe — 540 < 600)")
    print(f"  Warnings : {result8b['warnings']}")

    print("\n✓ All tests complete.")
