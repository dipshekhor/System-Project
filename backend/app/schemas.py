"""
schemas.py
==========
Day 11 · Pydantic Schemas (Request & Response Shapes)
-------------------------------------------------------
Pydantic schemas define the exact shape of:
  - REQUEST bodies  (what the frontend sends to the API)
  - RESPONSE bodies (what the API sends back)

Why separate schemas from ORM models?
  ORM models (models.py) define the DATABASE structure.
  Pydantic schemas define the API contract (what JSON looks like).
  They're intentionally separate because:
    - The DB may store more fields than you want to expose in the API
    - Requests don't include server-generated fields (id, created_at)
    - Responses may include computed fields not stored in DB
    - Validation logic belongs in the API layer, not the DB layer

Pydantic automatically:
  - Validates incoming JSON (returns 422 if fields are wrong type/missing)
  - Serializes ORM objects to JSON for responses (via from_attributes=True)
  - Generates the /docs Swagger UI schema automatically

Field validators:
  ge=1 means "greater than or equal to 1" (age can't be 0 or negative)
  le=120 means "less than or equal to 120"
  gt=0 means "greater than 0" (height/weight must be positive)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Profile schemas
# ──────────────────────────────────────────────────────────────────────────────

class ProfileCreate(BaseModel):
    """
    Schema for creating or updating a user profile.
    Sent by the frontend when user completes onboarding.

    Example request body:
        {
            "name": "Ahmed",
            "age": 45,
            "gender": "Male",
            "height_cm": 175,
            "weight_kg": 82,
            "diseases": ["Diabetes", "Hypertension"],
            "allergies": ["Nut Allergy"]
        }
    """
    name:       str   = Field(..., min_length=1, max_length=100,
                               description="User's full name")
    age:        int   = Field(..., ge=1, le=120,
                               description="Age in years")
    gender:     str   = Field(..., description="e.g. Male, Female, Other")
    height_cm:  float = Field(..., gt=0, le=300,
                               description="Height in centimetres")
    weight_kg:  float = Field(..., gt=0, le=700,
                               description="Weight in kilograms")

    # Optional medical fields
    # Empty list means no conditions / no allergies
    diseases:   list[str] = Field(default=[],
                                   description="Chronic diseases from: "
                                   "Diabetes, Hypertension, Heart Disease, "
                                   "Obesity, Kidney Disease")
    allergies:  list[str] = Field(default=[],
                                   description="Allergies from: "
                                   "Nut Allergy, Gluten Intolerance, Lactose Intolerance")

    # Optional lifestyle
    activity_level: Optional[str] = Field(None,
                                           description="e.g. Sedentary, Moderately Active, Very Active")
    dietary_pref:   Optional[str] = Field(None,
                                           description="e.g. Omnivore, Vegetarian, Vegan")

    @field_validator("diseases")
    @classmethod
    def validate_diseases(cls, v: list[str]) -> list[str]:
        """
        Validate that disease names match our supported set.
        Unknown diseases would be silently ignored by the rule engine,
        so we reject them early with a clear error message.
        """
        VALID = {
            "Diabetes", "Hypertension", "Heart Disease",
            "Obesity", "Kidney Disease", "None",
        }
        for disease in v:
            if disease not in VALID:
                raise ValueError(
                    f"Unknown disease: '{disease}'. "
                    f"Valid options: {sorted(VALID)}"
                )
        return v

    @field_validator("allergies")
    @classmethod
    def validate_allergies(cls, v: list[str]) -> list[str]:
        VALID = {"Nut Allergy", "Gluten Intolerance", "Lactose Intolerance", "None"}
        for allergy in v:
            if allergy not in VALID:
                raise ValueError(
                    f"Unknown allergy: '{allergy}'. "
                    f"Valid options: {sorted(VALID)}"
                )
        return v


class ProfileResponse(ProfileCreate):
    """
    Schema for returning a profile from the API.
    Extends ProfileCreate with server-generated fields (id, created_at).

    from_attributes=True: allows Pydantic to read values from SQLAlchemy
    ORM objects (not just dicts). Required for .model_validate(orm_obj).
    """
    id:         int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # read from SQLAlchemy ORM objects


# ──────────────────────────────────────────────────────────────────────────────
# Food check request schemas (one per input mode)
# ──────────────────────────────────────────────────────────────────────────────

class ManualFoodRequest(BaseModel):
    """
    Input Mode 3: User types a food name.

    Example:
        {"user_id": 1, "food_name": "grilled chicken salad"}
    """
    user_id:   int = Field(..., gt=0, description="User's profile ID")
    food_name: str = Field(..., min_length=1, max_length=300,
                            description="Food name to look up (typos are OK)")


class OCRTextRequest(BaseModel):
    """
    Input Mode 1 (Option A): Frontend already ran OCR, sends raw text.
    Use this when you do OCR client-side (e.g. ML Kit on mobile).

    Example:
        {
            "user_id": 1,
            "ocr_text": "Ingredients: wheat flour, sugar, salt, palm oil..."
        }
    """
    user_id:  int = Field(..., gt=0)
    ocr_text: str = Field(..., min_length=5, max_length=5000,
                           description="Raw OCR text from ingredient list photo")


class ImagePredictionRequest(BaseModel):
    """
    Input Mode 2: Teammate's food image model predicted a food label.
    Frontend calls teammate's model → gets label + confidence → sends here.

    Example:
        {
            "user_id": 1,
            "food_label": "Grilled Chicken Salad",
            "confidence": 0.91
        }
    """
    user_id:    int   = Field(..., gt=0)
    food_label: str   = Field(..., min_length=1, max_length=300,
                               description="Food label predicted by image model")
    confidence: float = Field(..., ge=0.0, le=1.0,
                               description="Model confidence score 0.0–1.0")


class ImageModelPredictionResponse(BaseModel):
    """
    Response returned by the image-classification model endpoint.

    Example:
        {
            "food_label": "spaghetti bolognese",
            "confidence": 0.91,
            "raw_food_label": "spaghetti_bolognese"
        }
    """
    food_label: str = Field(..., min_length=1, max_length=300)
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_food_label: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class NutrientInfo(BaseModel):
    """
    Nutrient breakdown shown to the user on the result screen.
    Optional — only present when a food was found in the database.
    """
    food_item:   str
    category:    Optional[str]  = None
    calories:    Optional[float] = None
    protein:     Optional[float] = None
    carbs:       Optional[float] = None
    fat:         Optional[float] = None
    fiber:       Optional[float] = None
    sugar:       Optional[float] = None
    sodium:      Optional[float] = None
    cholesterol: Optional[float] = None
    match_score: Optional[int]   = None   # fuzzy match confidence 0–100


class VerdictResponse(BaseModel):
    """
    The verdict result returned for every food check.
    This is what the frontend renders as the result screen.

    Example response:
        {
            "verdict": "avoid",
            "score": 30,
            "warnings": ["High sodium (1800mg) — dangerous for Hypertension"],
            "reasons": [],
            "food_info": {"food_item": "Instant Noodles", "sodium": 1800, ...},
            "ml_prediction": "avoid",
            "check_id": 42
        }
    """
    verdict:  str            = Field(..., description="safe | caution | avoid")
    score:    int            = Field(..., ge=0, le=100,
                                      description="Suitability score 0–100")
    warnings: list[str]      = Field(default=[])
    reasons:  list[str]      = Field(default=[])

    # Populated when a food was found in food_db_final_.csv
    food_info: Optional[dict] = Field(None,
                                       description="Nutrient profile of the matched food")

    # ML model's raw prediction (before blending with rules)
    ml_prediction: Optional[str]   = None
    ml_probabilities: Optional[dict] = None  # {'safe': 0.87, 'caution': 0.08, ...}

    # For OCR: list of ingredient→food matches
    ingredient_matches: Optional[list[dict]] = None

    # Image mode
    image_confidence: Optional[float] = None

    # BMI-derived info (populated when height/weight are in profile)
    bmi:      Optional[float] = None
    bmi_note: Optional[str]   = None

    # DB primary key of the saved FoodCheck row
    check_id: Optional[int] = None

    # ── Personalized nutrition engine output ──────────────────────────────────
    # user_targets:  predicted daily nutrient limits for THIS user
    #                {"target_calories": 2150, "target_sodium": 1500, ...}
    # budget_impact: fraction of each daily target this food consumes
    #                {"calories": 0.17, "sodium": 0.48, ...}
    user_targets:  Optional[dict] = None
    budget_impact: Optional[dict] = None


# ──────────────────────────────────────────────────────────────────────────────
# History schemas
# ──────────────────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    """
    One food check entry in the user's history list.
    A subset of FoodCheck columns — enough to show in a list view.
    """
    id:           int
    input_mode:   str
    query:        str
    food_found:   Optional[str]
    verdict:      str
    score:        int
    ml_prediction: Optional[str]
    checked_at:   datetime

    class Config:
        from_attributes = True


class HistoryDetail(HistoryItem):
    """
    Full food check details — shown when user taps a history item.
    Extends HistoryItem with warnings, reasons, and nutrient snapshot.
    """
    warnings:  list[str]
    reasons:   list[str]
    nutrients: Optional[dict]

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────────────────────────────────────
# Blog schemas
# ──────────────────────────────────────────────────────────────────────────────

class BlogPostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body:  str = Field(..., min_length=1, max_length=10000)


class BlogAnswerCreate(BaseModel):
    body: str = Field(..., min_length=1, max_length=2000)


class BlogAnswerOut(BaseModel):
    id:          int
    post_id:     int
    user_id:     int
    author_name: str
    body:        str
    created_at:  datetime


class BlogPostSummary(BaseModel):
    id:           int
    user_id:      int
    author_name:  str
    title:        str
    answer_count: int
    created_at:   datetime
    updated_at:   datetime


class BlogPostDetail(BaseModel):
    id:           int
    user_id:      int
    author_name:  str
    title:        str
    body:         str
    answer_count: int
    answers:      list[BlogAnswerOut]
    created_at:   datetime
    updated_at:   datetime
