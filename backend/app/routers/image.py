"""
routers/image.py
================
Day 14 · Food Image Analysis Endpoint
----------------------------------------
Handles Input Mode 2: user photographs food, teammate's model predicts label.

This router is intentionally thin — your teammate's model does the hard work
(image classification). This router just:
  1. Receives the predicted food label + confidence score
  2. Validates confidence (reject if too low)
  3. Looks up nutrients from food_db_final_.csv
  4. Runs hybrid_verdict
  5. Saves to DB and returns verdict

Integration pattern with teammate's model:
  Option A (Recommended): Each person runs their own server.
    - Teammate runs model server on port 5001
    - Frontend calls both: image model for label, then our API for verdict
    - No coupling between backends

  Option B: Chain internally
    - Frontend sends image to our API
    - Our API forwards image to teammate's model
    - Our API returns verdict
    See _forward_to_image_model() below for this pattern.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.schemas import ImagePredictionRequest, ImageModelPredictionResponse, VerdictResponse
from app.services import food_lookup, ml_model
from app.services import image_model_tflite
from app.services.medical_rules import hybrid_verdict
from app.routers.food import _get_profile_or_404, _save_check

router = APIRouter()


def _get_image_min_confidence() -> float:
    """
    Read minimum confidence from environment.
    Falls back to 0.20 and clamps to [0.0, 1.0].
    """
    raw = os.getenv("IMAGE_MIN_CONFIDENCE", "0.20").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.20
    return max(0.0, min(1.0, value))


@router.post(
    "/predict",
    summary="Predict food from uploaded image",
    description="Accepts image file, runs local TFLite inference, and returns food_name + confidence percentage.",
)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        pred = image_model_tflite.predict_from_image_bytes(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image prediction failed: {str(e)}")

    return {
        "food_name": pred["food_label"],
        "confidence_percentage": pred.get("confidence_percentage", round(float(pred["confidence"]) * 100.0, 2)),
        "confidence": pred["confidence"],
        "raw_food_label": pred.get("raw_food_label"),
        "normalized_food_label": pred.get("normalized_food_label"),
    }


@router.post(
    "/image-model/predict",
    response_model=ImageModelPredictionResponse,
    summary="Predict food label from photo using local TFLite model",
    description="Runs the teammate food-recognition .tflite model and returns {food_label, confidence}.",
)
async def predict_food_from_image(file: UploadFile = File(...)):
    """
    Teammate model endpoint for Part 4 flow:
      1) Frontend sends image
      2) Endpoint returns {food_label, confidence}
      3) Frontend calls POST /api/analyze-image with that response
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        pred = image_model_tflite.predict_from_image_bytes(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image prediction failed: {str(e)}")

    return ImageModelPredictionResponse(
        food_label=pred["food_label"],
        confidence=pred["confidence"],
        raw_food_label=pred.get("raw_food_label"),
    )


# ─── Option A: Frontend sends predicted label ─────────────────────────────────

@router.post(
    "/analyze-image",
    response_model=VerdictResponse,
    summary="Analyze food by image model prediction",
    description="Receives food label predicted by the image model "
                "and confidence score. Returns verdict based on the label.",
)
async def analyze_image(
    req: ImagePredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Input Mode 2: Food photo.

    The frontend flow:
      1. User takes photo
      2. Frontend sends photo to teammate's model → gets {food_label, confidence}
      3. Frontend sends {user_id, food_label, confidence} to this endpoint
      4. We look up nutrients, run verdict, return result

    Args:
        req.food_label:  e.g. "Grilled Chicken Salad", "Banana", "Pizza"
        req.confidence:  0.0–1.0 (reject if < IMAGE_MIN_CONFIDENCE)
    """
    min_confidence = _get_image_min_confidence()

    # ── Step 1: Reject low-confidence predictions ─────────────────────────────
    # If the image model confidence is very low, asking the user to retake
    # is better than giving a potentially wrong verdict.
    if req.confidence < min_confidence:
        raise HTTPException(
            status_code=400,
            detail=f"Image model confidence too low ({req.confidence:.0%}). "
                   f"Minimum required is {min_confidence:.0%}. "
                   f"Please retake the photo with better lighting or angle."
        )

    # ── Step 2: Load user profile ─────────────────────────────────────────────
    profile = await _get_profile_or_404(req.user_id, db)

    # ── Step 3: Look up nutrients ─────────────────────────────────────────────
    # Fuzzy-match the predicted label against food_db_final_.csv
    # If model predicts "Grilled Chicken Salad", this finds the exact DB row
    nutrients = food_lookup.lookup(req.food_label)
    if nutrients is None:
        raise HTTPException(
            status_code=404,
            detail=f"Predicted food '{req.food_label}' not found in database. "
                   f"The image model may have predicted something outside our database scope."
        )

    # ── Step 4: Build profile dict ────────────────────────────────────────────
    user_profile_dict = {
        "diseases":  profile.diseases,
        "allergies": profile.allergies,
    }

    # ── Step 5: Run hybrid verdict ─────────────────────────────────────────────
    result = hybrid_verdict(
        food_nutrients = nutrients,
        user_profile   = user_profile_dict,
        ml_predict_fn  = ml_model.predict,
    )

    # Also get ML probabilities
    disease_flags = {
        "has_" + d.lower().replace(" ", "_"): 1
        for d in profile.diseases
    }
    try:
        ml_probs = ml_model.predict_proba(nutrients, disease_flags)
    except Exception:
        ml_probs = None

    # ── Step 6: Save to DB ────────────────────────────────────────────────────
    check_id = await _save_check(
        db             = db,
        user_id        = req.user_id,
        mode           = "image",
        query          = req.food_label,
        food_found     = nutrients["food_item"],
        verdict_result = result,
        nutrients      = nutrients,
        confidence     = req.confidence,
    )

    # ── Step 7: Return verdict ────────────────────────────────────────────────
    return VerdictResponse(
        verdict          = result["verdict"],
        score            = result["score"],
        warnings         = result["warnings"],
        reasons          = result["reasons"],
        food_info        = nutrients,
        ml_prediction    = result.get("ml_prediction"),
        ml_probabilities = ml_probs,
        image_confidence = req.confidence,
        check_id         = check_id,
    )


# ─── Option B: Frontend sends raw image, we forward to teammate's model ───────

@router.post(
    "/analyze-food-photo",
    response_model=VerdictResponse,
    summary="Full pipeline: image upload → model prediction → verdict",
    description="Receives raw food photo, forwards to teammate's image model, "
                "then returns the verdict. Requires TEAMMATE_MODEL_URL env var.",
)
async def analyze_food_photo(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Option B: Full pipeline in one request.

    Frontend sends the photo here.
    This endpoint calls teammate's model internally, then runs verdict.

    Requires env var:
        TEAMMATE_MODEL_URL=http://localhost:5001

    If TEAMMATE_MODEL_URL is not set, returns a 503 error.
    """
    import os, httpx

    profile     = await _get_profile_or_404(user_id, db)
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    teammate_url = os.getenv("TEAMMATE_MODEL_URL", "")

    # ── Predict label via teammate URL OR local TFLite model ──────────────────
    if teammate_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{teammate_url}/predict",
                    files={"image": (file.filename, image_bytes, file.content_type)},
                )
                response.raise_for_status()
                model_result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Image model timed out (15s)")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Image model error: {str(e)}")
    else:
        try:
            model_result = image_model_tflite.predict_from_image_bytes(image_bytes)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image prediction failed: {str(e)}")

    # Expected response from teammate: {"food_label": "Pizza", "confidence": 0.91}
    food_label = model_result.get("food_label", "")
    confidence = float(model_result.get("confidence", 0.0))

    if not food_label:
        raise HTTPException(
            status_code=502,
            detail="Image model returned no food label. "
                   "Check teammate's model response format."
        )

    if confidence < 0.50:
        raise HTTPException(
            status_code=400,
            detail=f"Image model confidence too low ({confidence:.0%}). Retake photo."
        )

    # ── Look up nutrients ──────────────────────────────────────────────────────
    nutrients = food_lookup.lookup(food_label)
    if nutrients is None:
        raise HTTPException(
            status_code=404,
            detail=f"Predicted '{food_label}' not in food database."
        )

    user_profile_dict = {"diseases": profile.diseases, "allergies": profile.allergies}
    result = hybrid_verdict(nutrients, user_profile_dict, ml_model.predict)

    check_id = await _save_check(
        db             = db,
        user_id        = user_id,
        mode           = "image",
        query          = food_label,
        food_found     = nutrients["food_item"],
        verdict_result = result,
        nutrients      = nutrients,
        confidence     = confidence,
    )

    return VerdictResponse(
        verdict          = result["verdict"],
        score            = result["score"],
        warnings         = result["warnings"],
        reasons          = result["reasons"],
        food_info        = nutrients,
        ml_prediction    = result.get("ml_prediction"),
        image_confidence = confidence,
        check_id         = check_id,
    )
