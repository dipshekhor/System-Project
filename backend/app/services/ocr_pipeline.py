"""
ocr_pipeline.py
===============
Day 15–18 · OCR Pipeline for Ingredient List Photos
------------------------------------------------------
Converts a photo of food packaging into a clean ingredient string.

Pipeline:
  1. Read image bytes (from HTTP upload or file path)
  2. Preprocess: upscale → grayscale → denoise → adaptive threshold → deskew
  3. Run Tesseract OCR to extract raw text
  4. find_ingredients_section(): scan the text for the 'Ingredients:' block
  5. Return the ingredient text — passed to medical_rules via the /analyze-ocr router

Why preprocessing matters:
  Raw food packaging photos have: uneven lighting, glare, curved surfaces,
  small fonts, busy backgrounds. Without preprocessing, Tesseract accuracy
  drops from ~95% to ~50-60%. Each step targets a specific problem:
  
  - upscale     → Tesseract needs ≥150 DPI; phone photos of small text are often <80 DPI
  - grayscale   → removes color noise that confuses OCR
  - denoise     → removes JPEG compression artifacts  
  - threshold   → converts gray pixels to pure black/white (what OCR expects)
  - deskew      → corrects tilted text (common when photographing on a table)

Requirements:
  pip install pytesseract opencv-python-headless pillow
  # Linux: sudo apt install tesseract-ocr tesseract-ocr-eng
  # Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from pathlib import Path

from app.services.ocr_service import _run_google_vision

# Windows: set path explicitly so pytesseract finds Tesseract regardless of PATH.
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def preprocess_for_ocr(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR image array for optimal Tesseract OCR accuracy.
    
    Applies these steps in order:
      1. Upscale  — small images lose character details at OCR time
      2. Grayscale — OCR works on intensity, not colour
      3. Denoise  — removes JPEG/camera noise without blurring text edges
      4. Adaptive threshold — handles shadow/glare better than global threshold
      5. Deskew   — rotates the image to straighten any tilted text
    
    Args:
        image_array: BGR image as numpy array (from cv2.imread or cv2.imdecode)
    
    Returns:
        Preprocessed binary (black/white) image ready for Tesseract
    """
    # ── Step 1: Upscale if image is too small ────────────────────────────────
    # Tesseract accuracy drops sharply below ~150 DPI equivalent.
    # 1500px minimum gives better recognition of small label text.
    h, w = image_array.shape[:2]
    if w < 1500:
        scale  = 1500.0 / w
        new_w  = 1500
        new_h  = int(h * scale)
        image_array = cv2.resize(image_array, (new_w, new_h),
                                 interpolation=cv2.INTER_CUBIC)

    # ── Step 2: Convert to grayscale ─────────────────────────────────────────
    # OCR only needs intensity (brightness), not colour.
    # Grayscale also reduces noise from chromatic aberration.
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # ── Step 3: Denoise then sharpen ─────────────────────────────────────────
    # Denoise first to remove JPEG noise, then apply unsharp mask to
    # restore edge crispness lost during denoising (especially for blurry
    # phone photos where the text may be soft-focused).
    denoised = cv2.fastNlMeansDenoising(gray, h=7)
    _blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
    denoised = np.clip(
        cv2.addWeighted(denoised, 1.5, _blurred, -0.5, 0), 0, 255
    ).astype(np.uint8)

    # ── Step 4: Adaptive threshold ────────────────────────────────────────────
    # Converts each pixel to pure black or white.
    # ADAPTIVE_THRESH_GAUSSIAN_C adapts the threshold to local pixel neighbourhood,
    # which handles uneven lighting (one side of label bright, other dark).
    # blockSize=11: neighbourhood size (must be odd). C=2: subtracted from mean.
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # ── Step 5: Deskew ────────────────────────────────────────────────────────
    # Food labels photographed at an angle produce slanted text.
    # Clamp to ±15° — decorative elements (arrows, circles in educational
    # screenshots) can push minAreaRect to large wrong angles that flip
    # the entire image, making OCR output completely garbled.
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if 0.5 < abs(angle) < 15:
            (h2, w2) = thresh.shape
            center   = (w2 // 2, h2 // 2)
            M        = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh   = cv2.warpAffine(
                thresh, M, (w2, h2),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    return thresh


def extract_text_from_bytes(image_bytes: bytes) -> str:
    """
    Run full OCR pipeline on raw image bytes (from HTTP file upload).

    Tries Google Vision first; falls back to Tesseract if Vision is unavailable
    or returns no text.

    Args:
        image_bytes: raw bytes of the image file (JPEG, PNG, etc.)

    Returns:
        Extracted text string from the image

    Raises:
        ValueError: if image_bytes cannot be decoded as an image
    """
    vision_text = _run_google_vision(image_bytes)
    if vision_text and vision_text.strip():
        return vision_text.strip()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(
            "Could not decode image. Make sure the file is a valid JPEG/PNG."
        )

    return _run_tesseract(img)


def extract_text_from_path(image_path: str | Path) -> str:
    """
    Run full OCR pipeline on an image file path.

    Used for local testing — pass a file path string or Path object.

    Args:
        image_path: path to JPEG, PNG, or other image file

    Returns:
        Extracted text string
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    return extract_text_from_bytes(image_path.read_bytes())


def _run_tesseract(bgr_image: np.ndarray) -> str:
    """
    Internal helper: preprocess + run Tesseract on a BGR image array.

    Tries PSM 4 (single column) then PSM 6 (block), with bilingual eng+ben
    falling back to eng-only if Bengali tessdata is not installed.
    Picks the result with the most text content.
    """
    preprocessed = preprocess_for_ocr(bgr_image)

    best = ""
    for psm in (4, 6):
        for lang in ("eng+ben", "eng"):
            cfg = f"--oem 3 --psm {psm} --dpi 300"
            try:
                out = pytesseract.image_to_string(preprocessed, lang=lang, config=cfg)
            except pytesseract.TesseractError:
                continue
            if len(out) > len(best):
                best = out
            break  # if eng+ben succeeded, don't also try eng for same PSM

    return best.strip()


def find_ingredients_section(full_text: str) -> str:
    """
    Extract just the ingredients section from OCR output.
    
    Food packaging OCR produces a lot of noise: brand names, weight,
    nutritional facts table, allergen warnings, storage instructions, etc.
    We only want the ingredient list.
    
    Strategy:
      1. Scan for common ingredient section headers (case-insensitive)
      2. Once found, scan for the next section header to know where to stop
      3. Return the text between start and stop
      4. Fallback: return all text if no header found
    
    Args:
        full_text: raw OCR output from the full packaging image
    
    Returns:
        ingredient text block, or full_text as fallback
    
    Example:
        Input:  "Nutrition Facts ... Ingredients: wheat flour, sugar, salt.
                 Store in a cool dry place."
        Output: "Ingredients: wheat flour, sugar, salt."
    """
    text_lower = full_text.lower()

    # Patterns that signal the START of the ingredients section
    START_PATTERNS = [
        "ingredients:",
        "ingredients :",
        "ingredient list:",
        "contains:",
        "made with:",
        "composition:",
    ]

    # Patterns that signal the END of the ingredients section
    # (start of the next section we don't want)
    END_PATTERNS = [
        "nutrition facts",
        "nutritional information",
        "nutritional value",
        "allergen",
        "allergy advice",
        "may contain",
        "store in",
        "best before",
        "use by",
        "serving size",
        "% daily value",
        "%ri",
        "manufactured by",
        "distributed by",
        "net weight",
    ]

    # Find the start
    start_idx = -1
    for pattern in START_PATTERNS:
        idx = text_lower.find(pattern)
        if idx != -1:
            start_idx = idx
            break

    # If no ingredient header found, return the full text
    # (let the caller handle it — better to return too much than nothing)
    if start_idx == -1:
        return full_text

    # Extract from the start header onwards
    after_start = full_text[start_idx:]
    after_lower = after_start.lower()

    # Find the earliest end pattern
    end_idx = len(after_start)
    for pattern in END_PATTERNS:
        idx = after_lower.find(pattern)
        if idx != -1 and idx > 5:   # >5 to skip the start header itself
            end_idx = min(end_idx, idx)

    return after_start[:end_idx].strip()


def clean_ocr_text(raw_text: str) -> str:
    """
    Post-process raw OCR text to fix common OCR errors.
    
    Common OCR mistakes on food labels:
      - '0' (zero) misread as 'O' (letter) in amounts: "100g" → "1O0g"
      - '1' (one) misread as 'l' (lowercase L): "1%" → "l%"
      - Broken words due to hyphens at line ends: "wheat-\nflour" → "wheat flour"
      - Extra whitespace and newlines
    
    Args:
        raw_text: raw string from Tesseract
    
    Returns:
        Cleaned string
    """
    text = raw_text

    # Fix hyphenated line breaks (word split across lines)
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace multiple whitespace/newlines with single space
    text = re.sub(r"\s+", " ", text)

    # Remove common OCR noise characters (vertical bars, unusual punctuation)
    text = re.sub(r"[|¦§©®™°•·]", " ", text)

    # Remove percentage amounts (we don't need "contains 3% fat" type info)
    text = re.sub(r"\d+\.?\d*\s*%", "", text)

    # Remove E-numbers (food additives like E471, E322 — not in our food DB)
    text = re.sub(r"\bE\d{3,4}[a-z]?\b", "", text, flags=re.IGNORECASE)

    # Normalize spaces again after substitutions
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("ocr_pipeline.py — self-test (no real image needed)")
    print("=" * 55)

    # Test find_ingredients_section() with mock OCR text
    mock_ocr = """
    GOLDEN GRAIN CEREAL
    Net Weight: 500g
    
    Nutrition Facts
    Serving size: 30g
    Calories: 120
    
    Ingredients: Whole wheat flour, sugar, palm oil, 
    almonds (5%), skimmed milk powder, salt, 
    raising agent (E500), vanilla flavouring.
    
    Allergen Advice: Contains wheat, milk, nuts.
    Store in a cool dry place.
    Best before: see base of pack.
    """

    extracted = find_ingredients_section(mock_ocr)
    print(f"\nfind_ingredients_section() output:")
    print(f"  '{extracted}'")

    cleaned = clean_ocr_text(extracted)
    print(f"\nclean_ocr_text() output:")
    print(f"  '{cleaned}'")

    # Verify E-number removal
    assert "E500" not in cleaned, "E-numbers should be removed"
    print("\n✓ E-number removed")

    # Verify percentage removal
    assert "5%" not in cleaned, "Percentages should be removed"
    print("✓ Percentages removed")

    print("\n✓ All tests complete.")
    print("\nTo test with a real image:")
    print("  from ocr_pipeline import extract_text_from_path")
    print("  text = extract_text_from_path('your_food_label.jpg')")
    print("  ingredients = find_ingredients_section(text)")
    print("  print(clean_ocr_text(ingredients))")
