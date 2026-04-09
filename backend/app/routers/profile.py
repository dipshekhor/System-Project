"""
routers/profile.py
==================
Day 12 · Profile Endpoints
----------------------------
Handles creating, reading, and updating user medical profiles.

These endpoints are called:
  - POST /api/profile        → onboarding (user fills in name, diseases, allergies)
  - GET  /api/profile/{id}   → fetch profile (on app open, load saved profile)
  - PUT  /api/profile/{id}   → update profile (user edits their conditions)

The profile created here is passed as 'profile' to every food check.
The profile dict structure matches exactly what check_verdict() expects:
    {'diseases': [...], 'allergies': [...]}
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import UserProfile
from app.schemas import ProfileCreate, ProfileResponse

router = APIRouter()


@router.post(
    "/profile",
    response_model=ProfileResponse,
    status_code=201,
    summary="Create user profile",
    description="Create a new user profile with medical history. "
                "Returns the profile with server-assigned ID.",
)
async def create_profile(
    data: ProfileCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new user profile.

    Pydantic validates the request body before this function runs.
    If 'diseases' contains an invalid value, FastAPI returns 422 automatically.

    Steps:
      1. Create UserProfile ORM object from Pydantic data
      2. Add to session (not yet in DB)
      3. Flush → sends INSERT to DB, assigns auto-increment id
      4. Refresh → reloads the row from DB (picks up db-generated fields)
      5. Return as ProfileResponse (Pydantic serializes the ORM object)
    """
    # model_dump() converts Pydantic model → plain dict
    # Then ** unpacks it as kwargs to UserProfile constructor
    profile = UserProfile(**data.model_dump())

    db.add(profile)
    await db.flush()      # run INSERT, get id from DB
    await db.refresh(profile)  # re-read to pick up created_at, updated_at

    return profile


@router.get(
    "/profile/{user_id}",
    response_model=ProfileResponse,
    summary="Get user profile",
)
async def get_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch an existing user profile by ID.
    Returns 404 if the user doesn't exist.
    Called when the app opens to load the saved profile.
    """
    result  = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found with id={user_id}"
        )

    return profile


@router.put(
    "/profile/{user_id}",
    response_model=ProfileResponse,
    summary="Update user profile",
)
async def update_profile(
    user_id: int,
    data: ProfileCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Update an existing user profile.
    Replaces all fields with the new values from the request body.
    Returns 404 if user doesn't exist.

    Called when user edits their profile (adds a new disease, changes weight, etc.)
    The updated profile immediately affects future food checks.
    """
    result  = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found with id={user_id}"
        )

    # Update each field on the ORM object
    # setattr(profile, 'age', 46) is the same as profile.age = 46
    for key, value in data.model_dump().items():
        setattr(profile, key, value)

    await db.flush()
    await db.refresh(profile)
    return profile


@router.delete(
    "/profile/{user_id}",
    status_code=204,
    summary="Delete user profile",
)
async def delete_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a user profile and all their food check history.
    The cascade="all, delete-orphan" on the relationship handles
    deleting all related FoodCheck rows automatically.
    Returns 204 No Content on success (no response body).
    """
    result  = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile with id={user_id}")

    await db.delete(profile)  # cascade deletes FoodChecks too
