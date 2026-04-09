"""
models.py
=========
Day 10–11 · Database Table Definitions (SQLAlchemy ORM)
---------------------------------------------------------
Defines two tables:
  1. user_profiles  → stores medical history + demographics per user
  2. food_checks    → stores every food verdict check, linked to a user

Why ORM (Object-Relational Mapper)?
  Instead of writing raw SQL like:
    INSERT INTO user_profiles (name, age, diseases) VALUES (...)
  We write Python:
    profile = UserProfile(name="Ahmed", age=45, diseases=["Diabetes"])
    db.add(profile)
  SQLAlchemy translates this to SQL automatically.

Relationship:
  UserProfile ──< FoodCheck   (one user → many food checks)
  The FoodCheck.user_id column is a foreign key to UserProfile.id
  The 'cascade="all, delete-orphan"' means: if a user is deleted,
  all their food checks are also deleted automatically.

JSON columns:
  diseases, allergies, warnings, reasons, nutrients are stored as JSON.
  PostgreSQL has a native JSON type — SQLAlchemy's JSON column maps to it.
  This lets us store list/dict data without extra join tables.
"""

from datetime import datetime
from sqlalchemy import (
    String, Float, Integer, DateTime,
    JSON, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class UserProfile(Base):
    """
    Stores one row per app user.

    Created when the user completes the onboarding flow (enters name, age,
    diseases, allergies). Updated when they edit their profile.

    The 'diseases' and 'allergies' columns are JSON arrays, e.g.:
      diseases  = ["Diabetes", "Hypertension"]
      allergies = ["Nut Allergy"]

    These map directly to the lists expected by check_verdict() in medical_rules.py.
    """
    __tablename__ = "user_profiles"

    # ── Primary key ───────────────────────────────────────────────────────────
    # autoincrement=True (default for Integer PK) — DB assigns 1, 2, 3, ...
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Personal info ─────────────────────────────────────────────────────────
    name:       Mapped[str]   = mapped_column(String(100))
    age:        Mapped[int]   = mapped_column(Integer)
    gender:     Mapped[str]   = mapped_column(String(20))
    height_cm:  Mapped[float] = mapped_column(Float)
    weight_kg:  Mapped[float] = mapped_column(Float)
    email:           Mapped[str]      = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str]      = mapped_column(String(255))

    # ── Medical history ───────────────────────────────────────────────────────
    # Stored as JSON arrays. Example values from dataset:
    #   diseases  = ["Diabetes", "Hypertension", "Heart Disease", "Obesity", "Kidney Disease"]
    #   allergies = ["Nut Allergy", "Gluten Intolerance", "Lactose Intolerance"]
    diseases:  Mapped[list] = mapped_column(JSON, default=list)
    allergies: Mapped[list] = mapped_column(JSON, default=list)

    # ── Optional lifestyle fields ─────────────────────────────────────────────
    # nullable=True: these fields are optional during onboarding
    activity_level: Mapped[str | None] = mapped_column(String(50),  nullable=True)
    dietary_pref:   Mapped[str | None] = mapped_column(String(50),  nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    # created_at: set once when profile is created
    # updated_at: auto-updates on every save (onupdate=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # ── Relationship ──────────────────────────────────────────────────────────
    # 'checks' attribute gives access to all FoodCheck rows for this user.
    # cascade="all, delete-orphan": deleting a user also deletes all their checks.
    # lazy="selectin": SQLAlchemy loads checks with a separate SELECT (async-safe).
    checks: Mapped[list["FoodCheck"]] = relationship(
        "FoodCheck",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<UserProfile id={self.id} name={self.name!r} diseases={self.diseases}>"


class FoodCheck(Base):
    """
    Stores one row per food check performed by a user.
    This is the history log — one entry every time a user checks a food.

    input_mode values (3 input modes from the app):
      'manual' → user typed a food name
      'ocr'    → user scanned ingredient list photo
      'image'  → user took a food photo (teammate's model predicted label)

    The 'nutrients' column stores the full nutrient profile at check time
    as a JSON dict. This preserves exactly what was analyzed, even if
    food_db_final_.csv is updated later.
    """
    __tablename__ = "food_checks"

    # ── Primary key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Foreign key → UserProfile ─────────────────────────────────────────────
    # index=True: we frequently query "all checks for user X" — index speeds this up
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user_profiles.id", ondelete="CASCADE"),
        index=True,
    )

    # ── Input metadata ────────────────────────────────────────────────────────
    input_mode: Mapped[str]       = mapped_column(String(20))   # 'manual'|'ocr'|'image'
    query:      Mapped[str]       = mapped_column(String(500))  # what user typed/scanned
    food_found: Mapped[str | None]= mapped_column(String(200), nullable=True)  # matched food name

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict: Mapped[str] = mapped_column(String(10))  # 'safe' | 'caution' | 'avoid'
    score:   Mapped[int] = mapped_column(Integer)     # 0–100

    # JSON arrays of strings — warnings and positive reasons shown to user
    # e.g. warnings = ["High sodium — dangerous for Hypertension"]
    #      reasons  = ["Good fiber content", "Low sodium"]
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    reasons:  Mapped[list] = mapped_column(JSON, default=list)

    # ── Nutrient snapshot ─────────────────────────────────────────────────────
    # Full nutrient dict stored at check time.
    # e.g. {"calories": 180, "sodium": 180, "fat": 14, ...}
    # nullable=True: OCR checks may not resolve to a specific food
    nutrients: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── ML metadata (optional) ────────────────────────────────────────────────
    # Stores the raw ML prediction so we can audit it separately from final verdict
    ml_prediction: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence:    Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    # index=True: history queries sort by checked_at — index speeds ORDER BY
    checked_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )

    # ── Relationship back to user ─────────────────────────────────────────────
    user: Mapped["UserProfile"] = relationship(
        "UserProfile",
        back_populates="checks",
    )

    def __repr__(self) -> str:
        return (
            f"<FoodCheck id={self.id} user_id={self.user_id} "
            f"food={self.food_found!r} verdict={self.verdict!r}>"
        )
