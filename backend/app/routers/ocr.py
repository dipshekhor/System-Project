"""
routers/ocr.py
==============
Day 13–14 · OCR Ingredient Analysis Endpoints
-----------------------------------------------
Handles Input Mode 1: scanning the ingredient list on food packaging.

Two endpoint variants:
  POST /api/analyze-ocr       → frontend sends raw OCR text (text mode)
  POST /api/analyze-ocr-image → frontend sends image file, backend runs OCR

Why two variants?
  Mobile Option (faster): Run OCR on-device using ML Kit (Android) or
  Vision framework (iOS). The phone sends just the text string.
  → Use /api/analyze-ocr

  Web/Simple Option: Send the image file to the backend. Backend runs
  Tesseract OCR via ocr_pipeline.py.
  → Use /api/analyze-ocr-image

OCR-to-verdict flow:
  raw text
    → find_ingredients_section()   (isolate the 'Ingredients:' block)
    → clean_ocr_text()             (remove E-numbers, amounts, noise)
    → parse_ingredients()          (split into individual names)
    → food_lookup.lookup() each    (fuzzy-match each to food_db_final_.csv)
    → sum nutrients across matches (combined nutrient profile)
    → hybrid_verdict()             (rules + ML verdict)
    → save to DB → return
"""

import re
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import UserProfile
from app.schemas import OCRTextRequest, VerdictResponse
from app.services import food_lookup, ml_model
from app.services.medical_rules import hybrid_verdict
from app.services.ocr_pipeline import (
    extract_text_from_bytes,
    find_ingredients_section,
    clean_ocr_text,
)
from app.routers.food import _get_profile_or_404, _save_check, _enrich_result

router = APIRouter()


def parse_ingredients(text: str) -> list[str]:
    """
    Split an ingredient list string into individual ingredient names.

    Input:  "Whole wheat flour (45%), sugar, palm oil, almonds, salt, E471"
    Output: ["whole wheat flour", "sugar", "palm oil", "almonds", "salt"]

    Steps:
      1. Split on commas, semicolons, newlines (common ingredient separators)
      2. Remove percentage amounts (e.g. "45%" → "")
      3. Remove E-numbers (food additives not in our DB)
      4. Remove quantity strings (e.g. "100g", "2ml")
      5. Strip punctuation, lowercase, discard very short tokens
    """
    # Split on commas/semicolons/newlines
    parts = re.split(r"[,;\n]+", text)
    cleaned = []
    for part in parts:
        # Remove percentage values: "45%", "3.5%"
        part = re.sub(r"\b\d+\.?\d*\s*%", "", part)
        # Remove quantity+unit: "100g", "2ml", "50kcal"
        part = re.sub(r"\b\d+\.?\d*\s*(g|mg|ml|l|kcal|kj)\b", "", part, flags=re.I)
        # Remove E-numbers: E471, E322a
        part = re.sub(r"\bE\d{3,4}[a-z]?\b", "", part, flags=re.I)
        # Remove parenthetical sub-ingredients: (palm oil, water)
        # Keep text outside brackets as the primary ingredient name
        part = re.sub(r"\([^)]*\)", " ", part)
        # Strip punctuation, extra whitespace
        part = part.strip().strip("[]().,;:'\"").lower()
        # Keep only if it's a meaningful token (>2 chars)
        if len(part) > 2:
            cleaned.append(part)

    # Return max 30 ingredients — edge case on very long ingredient lists
    return cleaned[:30]


async def _process_ocr_text(
    ocr_text: str,
    user_id: int,
    profile: UserProfile,
    db: AsyncSession,
) -> VerdictResponse:
    """
    Shared logic for both OCR endpoints (text and image upload).
    Takes raw OCR text → returns VerdictResponse.

    This avoids duplicating the ingredient parsing + lookup + verdict logic.
    """
    # ── Step 1: Extract ingredients section ───────────────────────────────────
    # find_ingredients_section isolates the text after "Ingredients:" header
    # and stops at the next section (Nutrition Facts, Allergen Advice, etc.)
    ingredients_section = find_ingredients_section(ocr_text)
    cleaned_text        = clean_ocr_text(ingredients_section)

    # ── Step 2: Parse into individual ingredient names ────────────────────────
    ingredients = parse_ingredients(cleaned_text)

    if not ingredients:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any ingredients from the text. "
                   "Make sure the photo shows the ingredient list clearly."
        )

    # ── Step 3: Fuzzy-lookup each ingredient ──────────────────────────────────
    # We use a lower threshold (65) than manual input (70) because OCR
    # produces more imperfect text than user typing.
    combined_nutrients = {
        "calories": 0.0, "protein": 0.0, "carbs": 0.0,
        "fat": 0.0,      "fiber":   0.0, "sugar": 0.0,
        "sodium": 0.0,   "cholesterol": 0.0,
        # Pass the raw OCR text for allergy keyword scanning
        # This is more reliable than matching each ingredient to the DB
        # (the DB may not have every ingredient, but the allergy keywords
        # are scanned against the raw text directly in check_verdict)
        "ingredients_text": ocr_text.lower(),
    }

    ingredient_matches = []
    found_count = 0

    for ing in ingredients:
        nutrient = food_lookup.lookup(ing, threshold=65)
        if nutrient:
            found_count += 1
            # Sum nutrient values across all matched ingredients
            for key in ["calories", "protein", "carbs", "fat",
                        "fiber", "sugar", "sodium", "cholesterol"]:
                combined_nutrients[key] += float(nutrient.get(key, 0) or 0)

            ingredient_matches.append({
                "ingredient_query": ing,
                "food_found":       nutrient["food_item"],
                "match_score":      nutrient["match_score"],
            })
        else:
            # Record unmatched ingredients (shown in UI for transparency)
            ingredient_matches.append({
                "ingredient_query": ing,
                "food_found":       None,
                "match_score":      0,
            })

    # ── Step 4: Build user profile dict ──────────────────────────────────────
    user_profile_dict = {
        "diseases":         profile.diseases,
        "allergies":        profile.allergies,
        "ingredients_text": ocr_text.lower(),
        "age":              profile.age,
        "gender":           profile.gender,
        "height_cm":        profile.height_cm,
        "weight_kg":        profile.weight_kg,
    }

    # ── Step 5: Run hybrid verdict ─────────────────────────────────────────────
    result = hybrid_verdict(
        food_nutrients = combined_nutrients,
        user_profile   = user_profile_dict,
        ml_predict_fn  = ml_model.predict,
    )

    # ── Step 5b: Personalized daily-target context ───────────────────────────
    # Strip the non-numeric ingredients_text key before computing budget impact.
    nutrients_only = {k: v for k, v in combined_nutrients.items() if k != "ingredients_text"}
    _enrich_result(result, nutrients_only, profile)

    # Build the food_info shape the frontend ResultScreen expects
    food_info = {
        "food_item": f"Scanned ingredients ({found_count}/{len(ingredients)})",
        **nutrients_only,
    }

    # Attach ingredient match details to result
    result["ingredient_matches"] = ingredient_matches
    result["ingredients_found"]  = f"{found_count}/{len(ingredients)}"

    # ── Step 6: Save to DB ────────────────────────────────────────────────────
    check_id = await _save_check(
        db             = db,
        user_id        = user_id,
        mode           = "ocr",
        query          = ocr_text[:200],   # store first 200 chars of OCR text
        food_found     = None,             # OCR doesn't resolve to single food
        verdict_result = result,
        nutrients      = combined_nutrients,
    )

    return VerdictResponse(
        verdict             = result["verdict"],
        score               = result["score"],
        warnings            = result["warnings"],
        reasons             = result["reasons"],
        food_info           = food_info,
        ml_prediction       = result.get("ml_prediction"),
        bmi                 = result.get("bmi"),
        bmi_note            = result.get("bmi_note"),
        ingredient_matches  = ingredient_matches,
        check_id            = check_id,
        user_targets        = result.get("user_targets"),
        budget_impact       = result.get("budget_impact"),
    )


# ─── Endpoint A: Frontend sends raw OCR text ─────────────────────────────────

@router.post(
    "/analyze-ocr",
    response_model=VerdictResponse,
    summary="Analyze ingredients from OCR text",
    description="Receives raw OCR text (already extracted on the frontend). "
                "Parses ingredients, looks up nutrients, returns verdict.",
)
async def analyze_ocr_text(
    req: OCRTextRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Input Mode 1 (frontend OCR):
    Frontend uses ML Kit / Vision API to extract text, sends text here.
    Faster than image upload — no Tesseract needed on backend.
    """
    profile = await _get_profile_or_404(req.user_id, db)
    return await _process_ocr_text(req.ocr_text, req.user_id, profile, db)


# ─── Endpoint B: Frontend sends raw image, backend runs Tesseract ────────────

@router.post(
    "/analyze-ocr-image",
    response_model=VerdictResponse,
    summary="Analyze ingredient image (backend OCR)",
    description="Receives JPEG/PNG image of ingredient list. "
                "Backend runs Tesseract OCR, then analyses ingredients.",
)
async def analyze_ocr_image(
    user_id: int = Form(..., description="User profile ID"),
    file: UploadFile = File(..., description="Photo of food packaging ingredient list"),
    db: AsyncSession = Depends(get_db),
):
    """
    Input Mode 1 (backend OCR):
    User uploads a photo of the ingredient list.
    Backend runs the full OCR pipeline from ocr_pipeline.py.

    Note: UploadFile + Form requires python-multipart package (in requirements.txt).
    The user_id comes as a form field (not JSON) because we're uploading a file.
    """
    profile = await _get_profile_or_404(user_id, db)

    # Read image bytes from the upload
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Run Tesseract OCR pipeline (from ocr_pipeline.py)
    try:
        ocr_text = extract_text_from_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ocr_text.strip():
        raise HTTPException(
            status_code=422,
            detail="OCR extracted no text from the image. "
                   "Try better lighting or a clearer photo."
        )

    return await _process_ocr_text(ocr_text, user_id, profile, db)
