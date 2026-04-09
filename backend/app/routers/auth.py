# backend/app/routers/auth.py
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from jose import jwt
import os

from app.database import get_db
from app.models import UserProfile

router = APIRouter()

# Password hashing:
# - bcrypt_sha256 avoids bcrypt's 72-byte password limit
# - bcrypt is kept for backward compatibility with previously stored hashes
pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")

# JWT settings — reads from environment variable
SECRET_KEY  = os.getenv("SECRET_KEY", "change_this_in_production")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_DAYS = 30   # auto logout after 30 days


def hash_password(password: str) -> str:
    """Convert plain text password to bcrypt hash."""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Check plain password against stored hash."""
    return pwd_context.verify(plain, hashed)


def create_token(user_id: int) -> str:
    """
    Create a JWT token that expires in 30 days.
    Token contains: user_id + expiry time.
    After 30 days the token is invalid → user must log in again.
    """
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    payload = {
        "user_id": user_id,
        "exp":     expire,        # expiry — jose checks this automatically
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    """
    Decode JWT token and return user_id.
    Raises HTTPException if token is expired or invalid.
    """
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id  = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return user_id
    except Exception:
        raise HTTPException(status_code=401, detail="Token expired or invalid. Please log in again.")


# ── POST /api/auth/register ────────────────────────────────────────────────────
@router.post("/auth/register", status_code=201)
async def register(data: dict, db: AsyncSession = Depends(get_db)):
    """
    Create a new account.
    data: { email, password, name, age, gender, height_cm,
            weight_kg, diseases, allergies }
    Returns: { token, user_id, profile }
    """
    # Check email not already used
    result = await db.execute(
        select(UserProfile).where(UserProfile.email == data["email"].lower())
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered.")

    # Create profile with hashed password
    profile = UserProfile(
        email           = data["email"].lower().strip(),
        hashed_password = hash_password(data["password"]),
        name            = data["name"],
        age             = data["age"],
        gender          = data["gender"],
        height_cm       = data["height_cm"],
        weight_kg       = data["weight_kg"],
        diseases        = data.get("diseases", []),
        allergies       = data.get("allergies", []),
    )
    db.add(profile)
    await db.flush()
    await db.refresh(profile)

    token = create_token(profile.id)

    return {
        "token":    token,
        "user_id":  profile.id,
        "expires_in_days": TOKEN_EXPIRE_DAYS,
        "profile": {
            "id":        profile.id,
            "name":      profile.name,
            "email":     profile.email,
            "diseases":  profile.diseases,
            "allergies": profile.allergies,
        }
    }


# ── POST /api/auth/login ───────────────────────────────────────────────────────
@router.post("/auth/login")
async def login(data: dict, db: AsyncSession = Depends(get_db)):
    """
    Login with email + password.
    data: { email, password }
    Returns: { token, user_id, profile }
    """
    result = await db.execute(
        select(UserProfile).where(UserProfile.email == data["email"].lower())
    )
    profile = result.scalar_one_or_none()

    # Same error message for both wrong email and wrong password
    # Never tell attacker which one was wrong
    if not profile or not verify_password(data["password"], profile.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_token(profile.id)

    return {
        "token":    token,
        "user_id":  profile.id,
        "expires_in_days": TOKEN_EXPIRE_DAYS,
        "profile": {
            "id":        profile.id,
            "name":      profile.name,
            "email":     profile.email,
            "diseases":  profile.diseases,
            "allergies": profile.allergies,
        }
    }


# ── GET /api/auth/me ───────────────────────────────────────────────────────────
@router.get("/auth/me")
async def get_me(token: str, db: AsyncSession = Depends(get_db)):
    """
    Verify a saved token is still valid and return the profile.
    Called on app open to check if saved token has expired.
    token: passed as query param ?token=xxx
    Returns: profile dict or 401 if expired
    """
    user_id = decode_token(token)   # raises 401 if expired

    result = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "user_id": profile.id,
        "profile": {
            "id":        profile.id,
            "name":      profile.name,
            "email":     profile.email,
            "diseases":  profile.diseases,
            "allergies": profile.allergies,
        }
    }