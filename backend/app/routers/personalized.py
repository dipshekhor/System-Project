"""
routers/personalized.py
=======================
Personalized Nutrition endpoints.

  POST /api/personalized-targets   → predict the user's 8 daily nutrient targets
  POST /api/personalized-check     → look up a food and report budget impact
                                     against the user's PERSONAL daily limits

These complement the existing /api/check-food endpoint:
  - /check-food            → static medical-rule + ML safety verdict
  - /personalized-check    → "% of YOUR daily budget this food consumes"
                              tailored to the user's predicted targets
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routers.food import _get_profile_or_404
from app.services import personalized_nutrition

router = APIRouter()


class PersonalizedRequest(BaseModel):
    user_id:   int = Field(..., gt=0)
    food_name: str = Field(..., min_length=1, max_length=300)


class TargetsResponse(BaseModel):
    user_targets: dict


class PersonalizedVerdictResponse(BaseModel):
    food_item:      str
    food_nutrients: dict | None = None
    user_targets:   dict | None = None
    budget_impact:  dict | None = None
    verdict:        str
    score:          int
    reasons:        list[str]


def _profile_to_dict(profile) -> dict:
    return {
        "age":            profile.age,
        "gender":         profile.gender,
        "height":         profile.height_cm,
        "weight":         profile.weight_kg,
        "activity_level": getattr(profile, "activity_level", None),
        "diseases":       profile.diseases or [],
    }


@router.post(
    "/personalized-targets",
    response_model=TargetsResponse,
    summary="Predict the user's 8 daily nutrient targets",
)
async def personalized_targets(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    profile = await _get_profile_or_404(user_id, db)
    targets = personalized_nutrition.predict_user_targets(_profile_to_dict(profile))
    return TargetsResponse(user_targets=targets)


@router.post(
    "/personalized-check",
    response_model=PersonalizedVerdictResponse,
    summary="Check a food against the user's personal daily nutrient targets",
)
async def personalized_check(
    req: PersonalizedRequest,
    db: AsyncSession = Depends(get_db),
):
    profile = await _get_profile_or_404(req.user_id, db)
    result = personalized_nutrition.check_personalized_verdict(
        req.food_name, _profile_to_dict(profile)
    )
    if result["verdict"] == "Unknown":
        raise HTTPException(status_code=404, detail=result["reasons"][0])
    return PersonalizedVerdictResponse(**result)
