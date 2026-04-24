"""
food_lookup.py
==============
Day 5–6 · Food Database + Fuzzy Search
----------------------------------------
Loads food_db_final_.csv into memory once at startup and exposes
a single lookup() function that accepts any food name query —
including misspellings — and returns the full nutrient profile.

Why fuzzy matching?
  Users type things like "grilled chiken", "scrambl eggs", "choclate cake".
  Exact string matching fails for all of these. The fuzzywuzzy library
  computes a similarity score (0–100) using Levenshtein distance,
  so "griled chiken" still matches "Grilled Chicken Salad" at score ~85.

Module design:
  - load()   → called ONCE at startup (in FastAPI lifespan / notebook init)
  - lookup() → called on EVERY food check request
  
  This avoids re-reading the CSV on every request, which would be slow.
"""

import pandas as pd
from fuzzywuzzy import process
from pathlib import Path
import re

# ─── Module-level singletons ──────────────────────────────────────────────────
# These are set by load() and used by lookup().
# They are None until load() is called — lookup() will raise if called before load().
_food_db:    pd.DataFrame | None = None
_food_names: list[str]   | None = None

# Default data directory — override with load(data_dir=...) if needed
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# Synonym map: normalizes alternate names to the canonical DB food name (lowercase).
# This ensures image model labels and manual search always resolve to the same entry.
# Keys are post-normalization strings (lowercase, spaces only, no punctuation).
_SYNONYMS: dict[str, str] = {
    # hamburger → burger (image model predicts "hamburger"; DB canonical entry is "Burger")
    "hamburger": "burger",
    "beef burger": "burger",
    "cheeseburger": "burger",
    "cheese burger": "burger",
    # "Macaroni & Cheese" normalizes to "macaroni cheese" (& stripped); image model uses
    # "macaroni_and_cheese" → "macaroni and cheese" which hits "Macaroni And Cheese".
    # Unify both manual variants to the same canonical DB entry.
    "macaroni cheese": "macaroni and cheese",
    "mac and cheese": "macaroni and cheese",
    "mac cheese": "macaroni and cheese",
    # "Spring Roll" (singular) vs "Spring Rolls" (plural) are separate DB rows with
    # different nutritional data. Image model always predicts "spring_rolls" (plural),
    # so unify manual singular input to the same entry.
    "spring roll": "spring rolls",
}


def normalize_food_text(value: str) -> str:
        """
        Normalize food names so minor formatting differences do not affect matching.

        Examples:
            spaghetti_bolognese -> spaghetti bolognese
            Spaghetti-Bolognese -> spaghetti bolognese
            soy   beans         -> soy beans
        """
        if not value:
                return ""
        text = value.lower().strip()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def load(data_dir: Path | str | None = None) -> None:
    """
    Load and preprocess food_db_final_.csv into memory.
    Call this ONCE at startup before any lookup() calls.

    What this does:
      1. Reads the CSV
      2. Drops the empty trailing 'Unnamed: 12' column
      3. Renames columns to clean snake_case (e.g. 'Calories (kcal)' → 'calories_kcal')
      4. Adds a lowercased 'food_item_lower' column for fast case-insensitive matching
      5. Stores the DataFrame and name list in module globals

    Args:
        data_dir: path to the folder containing food_db_final_.csv
                  Defaults to backend/app/data/
    """
    global _food_db, _food_names

    dir_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    csv_path = dir_path / "food_db_final_.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"food_db_final_.csv not found at: {csv_path}\n"
            f"Make sure you copied it to backend/app/data/"
        )

    df = pd.read_csv(csv_path)

    # ── Drop the empty trailing column ───────────────────────────────────────
    # food_db_final_.csv has 'Unnamed: 12' (589 nulls out of 590 rows) — useless
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # ── Normalize column names to clean snake_case ────────────────────────────
    # Original: 'Calories (kcal)', 'Protein (g)', 'Sodium (mg)', etc.
    # After:    'calories_kcal',   'protein_g',   'sodium_mg',   etc.
    # This makes df['calories_kcal'] work instead of df['Calories (kcal)']
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s]+", "_", regex=True)    # spaces → underscore
        .str.replace(r"[()\/]", "", regex=True)     # remove ()/ 
        .str.replace(r"_+", "_", regex=True)        # collapse double underscores
        .str.strip("_")                              # strip leading/trailing _
    )

    # ── Add normalized version of food names for robust fuzzy matching ───────
    # This handles underscore/hyphen/punctuation differences automatically.
    df["food_item_search_key"] = df["food_item"].astype(str).map(normalize_food_text)

    _food_db    = df
    _food_names = df["food_item_search_key"].tolist()

    print(f"✓ Food database loaded: {len(_food_db)} items from {csv_path.name}")


def lookup(query: str, threshold: int = 70) -> dict | None:
    """
    Fuzzy-search the food database for a given food name.

    How it works:
      1. Lowercase + strip the query
      2. fuzzywuzzy.process.extractOne() finds the closest food name
         using WRatio (weighted ratio — handles partial matches well)
      3. If the best match score < threshold, return None (no match)
      4. Retrieve that row from the DataFrame and return as a clean dict

    Args:
        query:     food name to search for (e.g. "griled chiken", "oatmeal")
        threshold: minimum similarity score (0–100) to accept a match.
                   70 is a good default — accepts minor typos but rejects nonsense.
                   Lower = more lenient (more matches, more false positives)
                   Higher = stricter (fewer matches, fewer false positives)

    Returns:
        dict with nutrient keys, or None if no match found.

    Example:
        >>> lookup("scrambl eggs")
        {'food_item': 'Scrambled Eggs', 'calories': 180.0, 'sodium': 180.0, ...}

        >>> lookup("xyzabc123")
        None
    """
    if _food_db is None or _food_names is None:
        raise RuntimeError(
            "Food database not loaded. Call food_lookup.load() before lookup()."
        )

    if not query or not query.strip():
        return None

    query_clean = normalize_food_text(query)

    # Resolve synonyms before fuzzy matching so "hamburger" and "burger" always
    # hit the same DB row and produce the same verdict.
    query_clean = _SYNONYMS.get(query_clean, query_clean)

    # fuzzywuzzy.process.extractOne() returns (best_match_string, score)
    # WRatio is the default scorer — it handles partial matches, transpositions, etc.
    match, score = process.extractOne(query_clean, _food_names)

    # Reject if similarity is too low
    if score < threshold:
        return None

    # Retrieve the matching row from the DataFrame
    row = _food_db[_food_db["food_item_search_key"] == match].iloc[0]

    # Return a clean dict with standardized keys
    # These key names must match what medical_rules.py expects
    return {
        "food_item":        row["food_item"],           # original display name
        "category":         row.get("category", ""),
        "meal_type":        row.get("meal_type", ""),
        # ── Nutrients (float, default 0.0 for missing values) ─────────────────
        "calories":         float(row.get("calories_kcal", 0)),
        "protein":          float(row.get("protein_g", 0)),
        "carbs":            float(row.get("carbohydrates_g", 0)),
        "fat":              float(row.get("fat_g", 0)),
        "fiber":            float(row.get("fiber_g", 0)),
        "sugar":            float(row.get("sugars_g", 0)),
        "sodium":           float(row.get("sodium_mg", 0)),
        "cholesterol":      float(row.get("cholesterol_mg", 0)),
        "water_intake_ml":  float(row.get("water_intake_ml", 0)),
        # ── Match metadata (useful for debugging / UI display) ─────────────────
        "match_score":      score,                      # fuzzy similarity score
        "query_used":       query_clean,                # what the user typed
    }


def search_multiple(query: str, top_n: int = 5, threshold: int = 60) -> list[dict]:
    """
    Return the top N fuzzy matches for a query.
    Used on the frontend search screen to show disambiguation options when
    the user types something ambiguous (e.g. "chicken" → multiple results).

    Args:
        query:     search string
        top_n:     how many results to return
        threshold: minimum score to include

    Returns:
        list of dicts, each with {food_item, category, calories, match_score}
        sorted best-match first
    """
    if _food_db is None:
        raise RuntimeError("Food database not loaded. Call food_lookup.load() first.")

    query_clean = normalize_food_text(query)

    # process.extract() returns [(match_string, score), ...] for top N results
    matches = process.extract(query_clean, _food_names, limit=top_n)

    results = []
    for match_str, score in matches:
        if score < threshold:
            continue
        row = _food_db[_food_db["food_item_search_key"] == match_str].iloc[0]
        results.append({
            "food_item":   row["food_item"],
            "category":    row.get("category", ""),
            "calories":    float(row.get("calories_kcal", 0)),
            "match_score": score,
        })

    return results


def get_all_categories() -> list[str]:
    """Return sorted list of all food categories (for filtering in the app)."""
    if _food_db is None:
        raise RuntimeError("Food database not loaded.")
    return sorted(_food_db["category"].dropna().unique().tolist())


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("food_lookup.py — self-test")
    print("=" * 55)

    # Load from the data folder two levels up
    load(data_dir=Path(__file__).parent.parent / "data")

    # Test 1: Exact match
    r1 = lookup("Scrambled Eggs")
    print(f"\nTest 1 — Exact: 'Scrambled Eggs'")
    print(f"  Found    : {r1['food_item']}")
    print(f"  Calories : {r1['calories']} kcal")
    print(f"  Sodium   : {r1['sodium']} mg")
    print(f"  Score    : {r1['match_score']}")

    # Test 2: Typo
    r2 = lookup("griled chiken salad")
    print(f"\nTest 2 — Typo: 'griled chiken salad'")
    print(f"  Found    : {r2['food_item'] if r2 else 'NOT FOUND'}")
    print(f"  Score    : {r2['match_score'] if r2 else 'N/A'}")

    # Test 3: Partial
    r3 = lookup("oatmeal")
    print(f"\nTest 3 — Partial: 'oatmeal'")
    print(f"  Found    : {r3['food_item'] if r3 else 'NOT FOUND'}")

    # Test 4: No match
    r4 = lookup("xyzfakeitem999")
    print(f"\nTest 4 — No match: 'xyzfakeitem999'")
    print(f"  Found    : {r4}")   # should be None

    # Test 5: Top-5 search
    print(f"\nTest 5 — Top 5 matches for 'chicken':")
    for item in search_multiple("chicken", top_n=5):
        print(f"  {item['food_item']:40s} score={item['match_score']}")

    print("\n✓ All tests complete.")
