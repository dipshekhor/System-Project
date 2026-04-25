"""
main.py
=======
Day 11–12 · FastAPI Application Entry Point
---------------------------------------------
This is the root of the FastAPI application.

What this file does:
  1. Creates the FastAPI app instance
  2. Defines lifespan context manager (startup + shutdown logic)
  3. Attaches CORS middleware (allows frontend to call the API)
  4. Registers all routers (profile, food, ocr, image)
  5. Defines a /health endpoint for Docker health checks

Lifespan replaces the old @app.on_event("startup") pattern.
It runs setup code before the server starts accepting requests,
and teardown code after the server stops.

Startup order:
  1. Create DB tables (if not yet created by Alembic)
  2. Load food_lookup (reads food_db_final_.csv into memory)
  3. Load ml_model (reads .pkl files into memory)
  Both steps use module globals — they run ONCE, not per request.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth
from app.database import engine, Base
from app.routers import profile, food, ocr, image, blog, personalized
from app.services import food_lookup, ml_model, personalized_nutrition


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before accepting requests) and once at shutdown.

    Everything BEFORE 'yield' = startup code.
    Everything AFTER  'yield' = shutdown code.

    Why load food_lookup and ml_model here?
      Both involve reading files from disk (CSV and .pkl).
      If we loaded them inside each route handler, every request would
      re-read the files — very slow. Loading once at startup keeps them
      in memory for the lifetime of the server process.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────

    # Create DB tables for any models not yet in PostgreSQL.
    # Alembic handles schema migrations, but this ensures the tables
    # exist on first cold start even before running 'alembic upgrade head'.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables ready")

    # Load food database into memory (590 rows from food_db_final_.csv)
    # food_lookup._food_db and ._food_names become available globally
    food_lookup.load()

    # Load ML model from .pkl files
    # ml_model._model, ._label_encoder, ._feature_names become available
    try:
        ml_model.load()
    except FileNotFoundError:
        # Model .pkl files don't exist yet → run train first
        print("⚠ ML model not found. Training now (first run only)...")
        ml_model.train_and_save()
        ml_model.load()

    # Personalized nutrition engine (multi-output regressor → daily targets)
    try:
        personalized_nutrition.load()
    except FileNotFoundError:
        print("⚠ Personalized nutrition model not found. Training now...")
        personalized_nutrition.train_and_save()
        personalized_nutrition.load()

    print("✓ Server ready. Visit http://localhost:8000/docs")

    yield  # <-- server is live and accepting requests here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    # Dispose the connection pool gracefully on server stop
    await engine.dispose()
    print("Server shut down.")


# ─── Create FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Food Recommendation API",
    description=(
        "Tells users whether a food is safe to eat given their medical conditions. "
        "Supports 3 input modes: manual text, OCR ingredient scan, food photo."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Swagger UI at /docs  (enabled by default)
    # ReDoc UI at /redoc   (enabled by default)
)


# ─── CORS middleware ──────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) allows the frontend app to call this API
# even if it's running on a different port/domain (e.g. React Native on phone).
# allow_origins=["*"] allows ALL origins — fine for development.
# In production: replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: change to ["https://yourdomain.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)


# ─── Register routers ─────────────────────────────────────────────────────────
# Each router is defined in its own file in app/routers/.
# prefix="/api" means all endpoints start with /api
# e.g. profile router's POST / becomes POST /api/profile

app.include_router(profile.router, prefix="/api", tags=["Profile"])
app.include_router(food.router,    prefix="/api", tags=["Food Check"])
app.include_router(ocr.router,     prefix="/api", tags=["OCR"])
app.include_router(image.router,   prefix="/api", tags=["Image"])
app.include_router(auth.router, prefix="/api", tags=["Auth"])
app.include_router(blog.router, prefix="/api", tags=["Blog"])
app.include_router(personalized.router, prefix="/api", tags=["Personalized Nutrition"])



# ─── Health check endpoint ────────────────────────────────────────────────────
# Used by Docker's healthcheck in docker-compose.yml
# Also useful for monitoring tools (uptime checkers, etc.)
@app.get("/health", tags=["Health"])
async def health():
    """Returns 200 OK if the server is running."""
    return {"status": "ok", "version": "1.0.0"}
