"""
tests/test_api.py
=================
Day 27–28 · API Tests (pytest + httpx)
----------------------------------------
Tests all FastAPI endpoints end-to-end.

Uses pytest-asyncio for async test functions and httpx.AsyncClient
to make HTTP requests against the FastAPI app directly (no network).

Run:
    cd backend
    pytest app/tests/test_api.py -v

Or inside Docker:
    docker compose exec fastapi pytest app/tests/test_api.py -v

Test coverage:
  - Profile CRUD (create, read, update, not-found)
  - Manual food check (safe, caution, avoid, not-found)
  - OCR text analysis (normal, allergy trigger)
  - Image prediction (valid, low confidence)
  - History (get list, get detail, delete)

Important: These tests use a real SQLite in-memory database (not PostgreSQL)
to keep tests fast and self-contained. The DATABASE_URL is overridden via
dependency injection before each test module loads.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Override DATABASE_URL before importing app modules
import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"

from app.main import app
from app.database import Base, get_db
from app.services import food_lookup, ml_model


# ─── Test database setup ──────────────────────────────────────────────────────

# Use SQLite for tests (in-memory, no Docker required)
TEST_DB_URL = "sqlite+aiosqlite:///./test.db"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSession  = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    """Replace the real PostgreSQL session with a test SQLite session."""
    async with TestSession() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Override FastAPI's get_db dependency → use test DB
app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_db():
    """Create all tables in test DB before running any test in this module."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Load real services (food_lookup and ml_model use files, not DB)
    food_lookup.load()
    try:
        ml_model.load()
    except FileNotFoundError:
        ml_model.train_and_save()
        ml_model.load()

    yield  # run all tests

    # Cleanup: drop all tables after all tests in this module
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    """Async HTTP client that talks to the FastAPI app directly (no network)."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def sample_profile(client):
    """Create a test user profile with Hypertension + Nut Allergy. Returns profile dict."""
    resp = await client.post("/api/profile", json={
        "name":       "Test User",
        "age":        45,
        "gender":     "Male",
        "height_cm":  175,
        "weight_kg":  82,
        "diseases":   ["Hypertension"],
        "allergies":  ["Nut Allergy"],
    })
    assert resp.status_code == 201
    return resp.json()


@pytest_asyncio.fixture
async def clean_profile(client):
    """Create a test user with no diseases or allergies."""
    resp = await client.post("/api/profile", json={
        "name": "Healthy User", "age": 30, "gender": "Female",
        "height_cm": 165, "weight_kg": 60,
        "diseases": [], "allergies": [],
    })
    assert resp.status_code == 201
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Profile tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_profile(client):
    """Creating a profile returns 201 with an id."""
    resp = await client.post("/api/profile", json={
        "name": "Ahmed", "age": 40, "gender": "Male",
        "height_cm": 180, "weight_kg": 85,
        "diseases": ["Diabetes"], "allergies": [],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] > 0
    assert data["diseases"] == ["Diabetes"]


@pytest.mark.asyncio
async def test_get_profile(client, sample_profile):
    """Fetching an existing profile returns 200 with correct data."""
    uid  = sample_profile["id"]
    resp = await client.get(f"/api/profile/{uid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test User"
    assert "Hypertension" in data["diseases"]


@pytest.mark.asyncio
async def test_get_profile_not_found(client):
    """Fetching a non-existent profile returns 404."""
    resp = await client.get("/api/profile/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_profile(client, sample_profile):
    """Updating a profile with new diseases returns 200."""
    uid  = sample_profile["id"]
    resp = await client.put(f"/api/profile/{uid}", json={
        "name": "Test User", "age": 46, "gender": "Male",
        "height_cm": 175, "weight_kg": 83,
        "diseases": ["Hypertension", "Diabetes"],
        "allergies": ["Nut Allergy"],
    })
    assert resp.status_code == 200
    assert "Diabetes" in resp.json()["diseases"]


@pytest.mark.asyncio
async def test_invalid_disease_rejected(client):
    """Creating a profile with an invalid disease name returns 422."""
    resp = await client.post("/api/profile", json={
        "name": "Test", "age": 30, "gender": "Male",
        "height_cm": 170, "weight_kg": 70,
        "diseases": ["FakeDisease"],
        "allergies": [],
    })
    assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Manual food check tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_food_safe(client, clean_profile):
    """Scrambled eggs should be safe for a user with no conditions."""
    uid  = clean_profile["id"]
    resp = await client.post("/api/check-food", json={
        "user_id":   uid,
        "food_name": "Scrambled Eggs",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] in ("safe", "caution")  # eggs are generally safe
    assert data["score"] > 0
    assert data["check_id"] is not None            # saved to DB


@pytest.mark.asyncio
async def test_check_food_not_found(client, clean_profile):
    """Querying a nonsense food name returns 404."""
    resp = await client.post("/api/check-food", json={
        "user_id":   clean_profile["id"],
        "food_name": "xyzfakeitem999",
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_check_food_user_not_found(client):
    """Querying with a non-existent user_id returns 404."""
    resp = await client.post("/api/check-food", json={
        "user_id":   99999,
        "food_name": "Banana",
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_check_food_typo(client, clean_profile):
    """Fuzzy matching handles typos."""
    resp = await client.post("/api/check-food", json={
        "user_id":   clean_profile["id"],
        "food_name": "griled chiken salad",  # typo
    })
    # Should still find "Grilled Chicken Salad"
    assert resp.status_code == 200
    data = resp.json()
    assert data["food_info"] is not None


# ──────────────────────────────────────────────────────────────────────────────
# OCR tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ocr_normal(client, clean_profile):
    """Normal OCR text without allergens returns a verdict."""
    resp = await client.post("/api/analyze-ocr", json={
        "user_id":  clean_profile["id"],
        "ocr_text": "Ingredients: oats, brown sugar, corn flour, salt, vitamin D.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] in ("safe", "caution", "avoid")


@pytest.mark.asyncio
async def test_ocr_allergy_trigger(client, sample_profile):
    """OCR text with almonds should trigger Nut Allergy block."""
    resp = await client.post("/api/analyze-ocr", json={
        "user_id":  sample_profile["id"],
        "ocr_text": "Ingredients: wheat flour, sugar, almonds (5%), salt, palm oil.",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Allergy hard block → must be 'avoid' with score 0
    assert data["verdict"] == "avoid"
    assert data["score"] == 0
    assert any("ALLERGY" in w for w in data["warnings"])


@pytest.mark.asyncio
async def test_ocr_empty_text(client, clean_profile):
    """Too-short OCR text returns 422 (validation error)."""
    resp = await client.post("/api/analyze-ocr", json={
        "user_id":  clean_profile["id"],
        "ocr_text": "hi",   # min_length=5 validation
    })
    assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Image prediction tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_image_valid(client, clean_profile):
    """Valid food prediction with high confidence returns a verdict."""
    resp = await client.post("/api/analyze-image", json={
        "user_id":    clean_profile["id"],
        "food_label": "Banana",
        "confidence": 0.88,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] in ("safe", "caution", "avoid")
    assert data["image_confidence"] == 0.88


@pytest.mark.asyncio
async def test_image_low_confidence(client, clean_profile):
    """Image prediction with < 50% confidence returns 400."""
    resp = await client.post("/api/analyze-image", json={
        "user_id":    clean_profile["id"],
        "food_label": "Banana",
        "confidence": 0.35,   # below threshold
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_image_unknown_food(client, clean_profile):
    """Image model predicting a food not in our DB returns 404."""
    resp = await client.post("/api/analyze-image", json={
        "user_id":    clean_profile["id"],
        "food_label": "Exotic Unknown Dish XYZ",
        "confidence": 0.90,
    })
    assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# History tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_records_check(client, clean_profile):
    """After a food check, it appears in the user's history."""
    uid = clean_profile["id"]

    # Do a food check first
    await client.post("/api/check-food", json={
        "user_id": uid, "food_name": "Banana"
    })

    # Fetch history
    resp = await client.get(f"/api/history/{uid}")
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) >= 1
    assert any(h["input_mode"] == "manual" for h in history)


@pytest.mark.asyncio
async def test_history_detail(client, clean_profile):
    """Can fetch full detail for a specific check."""
    uid = clean_profile["id"]

    check_resp = await client.post("/api/check-food", json={
        "user_id": uid, "food_name": "Whole Wheat Toast"
    })
    check_id = check_resp.json()["check_id"]

    resp = await client.get(f"/api/history/{uid}/{check_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == check_id
    assert "nutrients" in detail   # full detail includes nutrient snapshot


@pytest.mark.asyncio
async def test_history_delete(client, clean_profile):
    """Deleting a check removes it from history."""
    uid = clean_profile["id"]

    check_resp = await client.post("/api/check-food", json={
        "user_id": uid, "food_name": "Scrambled Eggs"
    })
    check_id = check_resp.json()["check_id"]

    # Delete it
    del_resp = await client.delete(f"/api/history/{check_id}")
    assert del_resp.status_code == 204

    # Confirm it's gone
    gone = await client.get(f"/api/history/{uid}/{check_id}")
    assert gone.status_code == 404


# ─── Health check ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
