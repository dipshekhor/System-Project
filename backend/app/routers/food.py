"""
routers/food.py
===============
Day 12–13 · Manual Input + History Endpoints
----------------------------------------------
Handles Input Mode 3 (manual food name typing) and the history feature.

Endpoints:
  POST /api/check-food          → check a food by name
  GET  /api/history/{user_id}   → get user's check history
  GET  /api/history/{user_id}/{check_id} → get one check detail
  DELETE /api/history/{check_id} → delete a history entry

The _save_check() helper is shared with ocr.py and image.py —
all 3 input modes save to the same food_checks table.

Flow for POST /api/check-food:
  1. Validate request (Pydantic)
  2. Fetch user profile from DB
  3. Fuzzy-search food name in food_db_final_.csv (food_lookup.lookup)
  4. Run hybrid_verdict (rules + ML model)
  5. Save result to food_checks table
  6. Return VerdictResponse
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import UserProfile, FoodCheck
from app.schemas import ManualFoodRequest, VerdictResponse, HistoryItem, HistoryDetail
from app.services import food_lookup, ml_model
from app.services.medical_rules import hybrid_verdict

router = APIRouter()


# ─── Shared helpers (used by ocr.py and image.py too) ─────────────────────────

async def _get_profile_or_404(user_id: int, db: AsyncSession) -> UserProfile:
    """
    Fetch user profile from DB. Raise 404 if not found.
    Used by all 3 input mode routers to avoid repeating the same DB query.
    """
    result  = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"User profile with id={user_id} not found. "
                   f"Please create a profile first via POST /api/profile"
        )
    return profile


async def _save_check(
    db: AsyncSession,
    user_id: int,
    mode: str,          # 'manual' | 'ocr' | 'image'
    query: str,         # what the user typed / OCR text / image label
    food_found: str | None,
    verdict_result: dict,
    nutrients: dict | None,
    confidence: float | None = None,
) -> int:
    """
    Save a food check to the food_checks table. Returns the new check's id.

    This helper is called by all 3 routers after computing a verdict.
    Centralizing the save logic ensures consistent DB writes regardless of input mode.

    Args:
        mode:           'manual', 'ocr', or 'image'
        query:          truncated to 500 chars (column limit)
        food_found:     matched food name, or None if not found
        verdict_result: dict from hybrid_verdict()
        nutrients:      nutrient dict from food_lookup.lookup(), or None
        confidence:     image model confidence score (only for image mode)

    Returns:
        The auto-incremented id of the new FoodCheck row
    """
    check = FoodCheck(
        user_id    = user_id,
        input_mode = mode,
        query      = query[:500],        # truncate to column limit
        food_found = food_found,
        verdict    = verdict_result["verdict"],
        score      = verdict_result["score"],
        warnings   = verdict_result.get("warnings", []),
        reasons    = verdict_result.get("reasons", []),
        nutrients  = nutrients,
        ml_prediction = verdict_result.get("ml_prediction"),
        confidence = confidence,
    )
    db.add(check)
    await db.flush()     # run INSERT, get auto-generated id
    await db.refresh(check)
    return check.id


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/check-food",
    response_model=VerdictResponse,
    summary="Check food by name (manual input)",
    description="User types a food name. Fuzzy-matched against food_db_final_.csv, "
                "then rule + ML verdict returned.",
)
async def check_food(
    req: ManualFoodRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Input Mode 3: Manual text input.

    Step-by-step:
      1. Load user profile (diseases, allergies) from DB
      2. Fuzzy-match food name → get nutrient dict from food_db_final_.csv
      3. Build user_profile_dict for medical_rules
      4. hybrid_verdict: rules check → ML predict → blend scores
      5. Save FoodCheck to DB → get check_id
      6. Return VerdictResponse
    """
    # Step 1: Get user profile
    profile = await _get_profile_or_404(req.user_id, db)

    # Step 2: Fuzzy food lookup
    nutrients = food_lookup.lookup(req.food_name)
    if nutrients is None:
        raise HTTPException(
            status_code=404,
            detail=f"Food '{req.food_name}' not found in database. "
                   f"Try a simpler name or check spelling."
        )

    # Step 3: Build profile dict (matches medical_rules.check_verdict signature)
    user_profile_dict = {
        "diseases":  profile.diseases,   # list from JSON column
        "allergies": profile.allergies,  # list from JSON column
    }

    # Step 4: Run hybrid verdict (rules + ML)
    # ml_model.predict is passed as a callable — medical_rules calls it internally
    result = hybrid_verdict(
        food_nutrients  = nutrients,
        user_profile    = user_profile_dict,
        ml_predict_fn   = ml_model.predict,
    )

    # Also get ML probability breakdown for UI display
    disease_flags = {
        "has_" + d.lower().replace(" ", "_"): 1
        for d in profile.diseases
    }
    try:
        ml_probs = ml_model.predict_proba(nutrients, disease_flags)
    except Exception:
        ml_probs = None

    # Step 5: Save to DB
    check_id = await _save_check(
        db         = db,
        user_id    = req.user_id,
        mode       = "manual",
        query      = req.food_name,
        food_found = nutrients["food_item"],
        verdict_result = result,
        nutrients  = nutrients,
    )

    # Step 6: Return verdict
    return VerdictResponse(
        verdict           = result["verdict"],
        score             = result["score"],
        warnings          = result["warnings"],
        reasons           = result["reasons"],
        food_info         = nutrients,
        ml_prediction     = result.get("ml_prediction"),
        ml_probabilities  = ml_probs,
        check_id          = check_id,
    )


@router.get(
    "/history/{user_id}",
    response_model=list[HistoryItem],
    summary="Get user's food check history",
)
async def get_history(
    user_id: int,
    limit:   int = 20,    # max rows to return (default 20, max 100)
    db: AsyncSession = Depends(get_db),
):
    """
    Return a paginated list of the user's recent food checks.
    Results are ordered newest-first (checked_at DESC).
    Used to render the history screen in the app.

    Query params:
        limit: max records to return (1–100, default 20)
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    await _get_profile_or_404(user_id, db)  # 404 if user doesn't exist

    result = await db.execute(
        select(FoodCheck)
        .where(FoodCheck.user_id == user_id)
        .order_by(FoodCheck.checked_at.desc())
        .limit(limit)
    )
    checks = result.scalars().all()
    return checks  # Pydantic serializes each via HistoryItem.from_attributes


@router.get(
    "/history/{user_id}/{check_id}",
    response_model=HistoryDetail,
    summary="Get one food check detail",
)
async def get_check_detail(
    user_id:  int,
    check_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Return the full detail of one food check (including nutrients, warnings, reasons).
    Called when user taps an item in their history list.
    """
    result = await db.execute(
        select(FoodCheck).where(
            FoodCheck.id == check_id,
            FoodCheck.user_id == user_id,   # security: user can only see their own checks
        )
    )
    check = result.scalar_one_or_none()

    if check is None:
        raise HTTPException(
            status_code=404,
            detail=f"Check id={check_id} not found for user id={user_id}"
        )
    return check


@router.delete(
    "/history/{check_id}",
    status_code=204,
    summary="Delete a food check from history",
)
async def delete_check(
    check_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a single food check entry from history.
    Called when user swipes to delete in the history list.
    Returns 204 No Content (no response body).
    """
    await db.execute(
        delete(FoodCheck).where(FoodCheck.id == check_id)
    )
