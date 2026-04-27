"""
ocr_service.py
==============
Two-tier OCR for nutrition-fact label parsing (English + Bangla).

Design goal: stable extraction with simple rules.
- Match nutrient keywords directly.
- Read the immediate next number after keyword on the same line.
- Require unit for non-calorie nutrients.
- If calories has no unit, treat as kcal.
"""

from __future__ import annotations

import os
import re
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# Windows: set path explicitly so pytesseract finds Tesseract regardless of PATH.
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

try:
    from google.cloud import vision  # type: ignore
    _VISION_AVAILABLE = True
except Exception:  # pragma: no cover
    _VISION_AVAILABLE = False


log = logging.getLogger(__name__)


NUTRIENT_KEYWORDS: dict[str, list[str]] = {
    "calories": ["calories", "calorie", "cal0ries", "ca1ories", "calorles", "caiories", "energy", "kcal", "ক্যালরি", "ক্যালোরি", "শক্তি"],
    "protein": ["protein", "আমিষ", "প্রোটিন"],
    "carbs": ["carbohydrate", "carbohydrates", "carbs", "শর্করা", "কার্বোহাইড্রেট"],
    "fat": ["fat", "lipid", "total fat", "চর্বি", "স্নেহ"],
    "sugar": ["sugar", "sugars", "চিনি", "সুগার"],
    "sodium": ["sodium", "sodlum", "sodiurn", "sodim", "salt", "লবণ", "সোডিয়াম"],
    "cholesterol": ["cholesterol", "choiesterol", "cholesteroi", "choiesteroi", "কোলেস্টেরল"],
    "fiber": ["fiber", "fibre", "dietary fiber", "আঁশ", "ফাইবার"],
}

_BN_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_UNIT_TO_FACTOR_G = {
    "g": 1.0,
    "gm": 1.0,
    "গ্রাম": 1.0,
    "গ্রা": 1.0,   # Bangla abbreviated gram (গ্রা: is common on local labels)
    "mg": 0.001,
    "মিগ্রা": 0.001,
    "মিলিগ্রাম": 0.001,
    "kg": 1000.0,
}
_UNIT_TO_FACTOR_MG = {
    "mg": 1.0,
    "মিগ্রা": 1.0,
    "মিলিগ্রাম": 1.0,
    "g": 1000.0,
    "gm": 1000.0,
    "গ্রাম": 1000.0,
    "গ্রা": 1000.0,  # Bangla abbreviated gram
}

_NUM_ANY_RE = re.compile(
    r"([0-9]+(?:[.,][0-9]+)?)\s*"
    r"(kcal|kj|mg|g|gm|kg|গ্রাম|গ্রা|মিগ্রা|মিলিগ্রাম|কিজু)?",
    re.IGNORECASE,
)

_SECTION_VALUE_HINT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kcal|kj|mg|g|gm|kg|গ্রাম|গ্রা|মিগ্রা|মিলিগ্রাম|কিজু)\b",
    re.IGNORECASE,
)

_URL_NOISE_RE = re.compile(
    r"(?:https?://|www\.|\.com\b|search\?|google\.|chrome|youtube|facebook)",
    re.IGNORECASE,
)

# Words/patterns that indicate a number is a serving-size measure, not a calorie count.
# /\d catches fraction numerators like "2" in "2/3 cup".
_SERVING_UNIT_RE = re.compile(
    r"^\s*(?:/\d|cups?|tbsp|tsp|fl\.?\s*oz|oz|ml|m[Ll]|litr[es]?|servings?|pieces?|slices?|packets?)",
    re.IGNORECASE,
)

def _run_google_vision(image_bytes: bytes) -> str | None:
    global _VISION_AVAILABLE, vision

    if not _VISION_AVAILABLE:
        try:
            from google.cloud import vision as _vision  # type: ignore
            vision = _vision
            _VISION_AVAILABLE = True
        except Exception:
            return None

    cred_path = _ensure_google_credentials_env()
    if not cred_path:
        return None
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        ctx = vision.ImageContext(language_hints=["bn", "en"])
        resp = client.document_text_detection(image=image, image_context=ctx)
        if resp.error.message:
            log.warning("Vision API error: %s", resp.error.message)
            return None
        if resp.full_text_annotation and resp.full_text_annotation.text:
            return resp.full_text_annotation.text

        # Some label photos return better results with text_detection.
        resp2 = client.text_detection(image=image, image_context=ctx)
        if resp2.error.message:
            log.warning("Vision text_detection error: %s", resp2.error.message)
            return None
        if resp2.full_text_annotation and resp2.full_text_annotation.text:
            return resp2.full_text_annotation.text

        if resp2.text_annotations:
            return resp2.text_annotations[0].description

        return None
    except Exception as exc:  # pragma: no cover
        log.warning("Google Vision call failed: %s", exc)
        return None


def _ensure_google_credentials_env() -> str | None:
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    cred_dir = Path(__file__).resolve().parents[2] / "credentials"
    if not cred_dir.exists():
        return None

    json_files = sorted(cred_dir.glob("*.json"))
    if not json_files:
        return None

    detected = str(json_files[0])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = detected
    return detected


def _deskew(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return binary
    h, w = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        binary,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    h, w = img.shape[:2]
    if w < 1200:
        scale = 1200.0 / w
        img = cv2.resize(img, (1200, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def _preprocess_variants(image_bytes: bytes) -> list[np.ndarray]:
    img = _decode_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = denoised.shape[:2]
    y0, y1 = int(h * 0.08), int(h * 0.92)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    crop = denoised[y0:y1, x0:x1]

    variants = [_deskew(adaptive), _deskew(otsu), denoised]
    if crop.size > 0:
        crop_adaptive = cv2.adaptiveThreshold(
            crop,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        variants.append(_deskew(crop_adaptive))
    return variants


def _keyword_hit_count(text: str) -> int:
    low = text.lower()
    hits = 0
    for keywords in NUTRIENT_KEYWORDS.values():
        for kw in keywords:
            if kw.lower() in low:
                hits += 1
                break
    return hits


def _normalise_value(num_str: str, unit: str | None, target_key: str) -> float:
    num = float(num_str)
    if unit is None:
        return num
    unit = unit.strip().lower()

    if target_key in ("sodium", "cholesterol"):
        return num * _UNIT_TO_FACTOR_MG.get(unit, 1.0)
    if target_key in ("protein", "fat", "carbs", "fiber", "sugar"):
        return num * _UNIT_TO_FACTOR_G.get(unit, 1.0)
    if target_key == "calories":
        if unit in ("kj", "কিজু"):
            return num / 4.184
        return num
    return num


def _is_allowed_unit(target_key: str, unit: str) -> bool:
    unit = unit.strip().lower()
    if target_key == "calories":
        return unit in ("kcal", "kj", "কিজু")
    if target_key in ("sodium", "cholesterol"):
        return unit in ("mg", "g", "gm", "kg", "মিগ্রা", "মিলিগ্রাম", "গ্রাম", "গ্রা")
    if target_key in ("protein", "fat", "carbs", "fiber", "sugar"):
        return unit in ("g", "gm", "mg", "kg", "গ্রাম", "গ্রা", "মিগ্রা", "মিলিগ্রাম")
    return True


def _extract_first_number(window: str, target_key: str) -> float | None:
    # Percentage figures (e.g., Daily Value 10%) are noise for nutrient value extraction.
    clean = re.sub(r"\b\d+(?:[.,]\d+)?\s*%", "", window)

    # OCR often reads zero as letter O in nutrient lines (e.g., "Omg" instead of "0mg").
    clean = re.sub(r"\b[Oo](?=\s*(?:kcal|kj|mg|g|gm|kg|গ্রাম|গ্রা|মিগ্রা|মিলিগ্রাম|কিজু)\b)", "0", clean)
    clean = re.sub(r"\b[Oo](?=[.,]?\d)", "0", clean)

    if target_key == "calories":
        # Normalize common OCR confusions inside numeric calorie values (e.g., "23O" -> "230").
        clean = re.sub(r"(?<=\d)[Oo](?=\d|\b)", "0", clean)
        clean = re.sub(r"(?<=\d)[Il](?=\d|\b)", "1", clean)
        clean = re.sub(r"(?<=\d)S(?=\d|\b)", "5", clean)
        clean = re.sub(r"(?<=\d)B(?=\d|\b)", "8", clean)

    for m in _NUM_ANY_RE.finditer(clean):
        raw_num = m.group(1).replace(",", ".")
        raw_unit = m.group(2)

        if target_key == "calories" and not raw_unit:
            # Skip numbers that are followed by a serving-size word (e.g. "1 cup", "2 tbsp").
            # Those are serving-size measurements, not calorie counts.
            if _SERVING_UNIT_RE.match(clean[m.end():]):
                continue
            raw_unit = "kcal"

        if target_key != "calories" and not raw_unit:
            # OCR often drops units for sodium/cholesterol; treat as mg by default.
            if target_key in ("sodium", "cholesterol"):
                raw_unit = "mg"
            elif target_key in ("protein", "fat", "carbs", "sugar", "fiber"):
                # Bangla table layouts often use ":" as a cell separator with no unit.
                # If the number is followed only by ":" (or end of text), treat as grams.
                rest = clean[m.end():].lstrip()
                if rest.startswith(":") or rest == "":
                    raw_unit = "g"
                else:
                    continue
            else:
                continue
        if raw_unit and not _is_allowed_unit(target_key, raw_unit):
            continue

        try:
            value = _normalise_value(raw_num, raw_unit, target_key)
        except ValueError:
            continue
        return value

    return None


def _strip_to_section(text: str) -> str:
    headers = [
        "nutrition facts",
        "nutrition information",
        "nutritional information",
        "nutritional value",
        "p nutrition",
        "nutrition",
        "পুষ্টি তথ্য",
        "পুষ্টি বিষয়ক তথ্য",
        "পুষ্টি",
        "পুষ্টিগুণ",
    ]
    low = text.lower()

    # Browser screenshots often contain earlier "nutrition" text in URL/tab chrome.
    # Pick the section whose following window looks most like a nutrient table.
    candidates: list[int] = []
    for h in headers:
        pos = 0
        while True:
            idx = low.find(h, pos)
            if idx == -1:
                break
            candidates.append(idx)
            pos = idx + 1

    if not candidates:
        return text

    best_idx = candidates[0]
    best_score = -1
    for idx in candidates:
        window = text[idx: idx + 1400]
        score = (_keyword_hit_count(window) * 5) + len(_SECTION_VALUE_HINT_RE.findall(window))
        if score > best_score:
            best_score = score
            best_idx = idx

    return text[best_idx:]


def _is_noise_line(line: str) -> bool:
    if _URL_NOISE_RE.search(line):
        return True

    # Drop browser/tab lines that are mostly ASCII words and punctuation.
    alpha = sum(ch.isalpha() for ch in line)
    if alpha == 0:
        return False
    ascii_alpha = sum(ch.isascii() and ch.isalpha() for ch in line)
    return (ascii_alpha / alpha) > 0.9 and ("nutrition" in line.lower() or "search" in line.lower())


def _find_keyword_index(line: str, keywords: list[str]) -> tuple[int, str] | None:
    line_low = line.lower()
    best_idx = -1
    best_kw = ""
    for kw in keywords:
        kw_low = kw.lower()
        for m in re.finditer(re.escape(kw_low), line_low):
            start = m.start()
            end = m.end()
            prev_ok = start == 0 or not line_low[start - 1].isalpha()
            next_ok = end == len(line_low) or not line_low[end].isalpha()
            if not (prev_ok and next_ok):
                continue
            if best_idx == -1 or start < best_idx:
                best_idx = start
                best_kw = kw
            break
    if best_idx == -1:
        return None
    return best_idx, best_kw


def parse_nutrients(raw_text: str) -> dict[str, float]:
    if not raw_text:
        return {}

    text = _strip_to_section(raw_text).translate(_BN_DIGIT_MAP)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not _is_noise_line(ln.strip())]
    found: dict[str, float] = {}
    keyword_positions: dict[str, int] = {}  # nutrient -> line index where keyword found

    for i, line in enumerate(lines):
        for nutrient, keywords in NUTRIENT_KEYWORDS.items():
            if nutrient in found:
                continue
            hit = _find_keyword_index(line, keywords)
            if not hit:
                continue
            best_idx, best_kw = hit
            after_kw = line[best_idx + len(best_kw):]
            # Skip "Calories from Fat" lines — we want total calories only.
            if nutrient == "calories" and re.match(r"\s*from\b", after_kw, re.IGNORECASE):
                continue
            keyword_positions.setdefault(nutrient, i)
            # OCR may split "Calories" and "230" across lines. Try nearby windows.
            candidate_windows = [after_kw]
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                candidate_windows.append(nxt)
                candidate_windows.append(f"{after_kw} {nxt}")
            if nutrient == "calories" and i + 2 < len(lines):
                nxt2 = lines[i + 2]
                candidate_windows.append(f"{after_kw} {lines[i + 1]} {nxt2}")

            for window in candidate_windows:
                value = _extract_first_number(window, nutrient)
                if value is not None:
                    found[nutrient] = round(value, 2)
                    break

    # Column-separated table fallback: Google Vision sometimes outputs a table
    # column-by-column, putting all values BEFORE all keywords. Collect every
    # value-with-unit token from the full text in order, then assign leftover
    # values to leftover keywords by the order their keywords first appeared.
    missing = [k for k in NUTRIENT_KEYWORDS if k not in found and k in keyword_positions]
    if missing:
        used_values: set[float] = {round(v, 2) for v in found.values()}
        all_tokens: list[tuple[float, str]] = []  # (value, unit)
        for m in re.finditer(
            r"([0-9]+(?:[.,][0-9]+)?)\s*"
            r"(kcal|kj|mg|g|gm|kg|গ্রাম|গ্রা|মিগ্রা|মিলিগ্রাম|কিজু)",
            text,
            re.IGNORECASE,
        ):
            try:
                num = float(m.group(1).replace(",", "."))
            except ValueError:
                continue
            unit = m.group(2) or ""
            all_tokens.append((num, unit))

        # Sort missing nutrients by where their keyword appeared in the text.
        missing_sorted = sorted(missing, key=lambda k: keyword_positions[k])
        for nutrient in missing_sorted:
            for idx, (num, unit) in enumerate(all_tokens):
                if not unit:
                    continue
                if not _is_allowed_unit(nutrient, unit):
                    continue
                try:
                    value = round(_normalise_value(str(num), unit, nutrient), 2)
                except ValueError:
                    continue
                if value in used_values:
                    continue
                # Plausibility: reject obviously wrong magnitudes per nutrient.
                if nutrient == "calories" and (value < 1 or value > 2000):
                    continue
                if nutrient in ("protein", "fat", "carbs", "fiber", "sugar") and value > 200:
                    continue
                if nutrient == "sodium" and value > 5000:
                    continue
                found[nutrient] = value
                used_values.add(value)
                all_tokens.pop(idx)
                break

    # Last resort for calories in noisy OCR where line matching failed.
    if "calories" not in found:
        low_text = text.lower()
        cal_fallback_re = re.compile(
            r"(?:calories|calorie|cal0ries|ca1ories|calorles|caiories|energy|kcal|ক্যালরি|ক্যালোরি|শক্তি)"
            r"(?!\s*from\b)"
            r"[^0-9]{0,40}([0-9]+(?:[.,][0-9]+)?)",
            re.IGNORECASE,
        )
        m = cal_fallback_re.search(low_text)
        if m:
            try:
                found["calories"] = round(float(m.group(1).replace(",", ".")), 2)
            except ValueError:
                pass

    return found


def _run_tesseract_multi(image_bytes: bytes) -> tuple[str, dict[str, float]]:
    variants = _preprocess_variants(image_bytes)

    def _ocr(img: np.ndarray, psm: int, lang: str) -> str:
        config = f"--oem 3 --psm {psm}"
        try:
            return pytesseract.image_to_string(img, lang=lang, config=config)
        except pytesseract.TesseractError:
            if lang != "eng":
                return pytesseract.image_to_string(img, lang="eng", config=config)
            return ""

    texts: list[tuple[str, int]] = []
    parses: list[dict[str, float]] = []
    for img in variants:
        for psm, lang in ((4, "eng+ben"), (6, "eng+ben"), (11, "eng+ben"), (4, "eng"), (6, "eng"), (11, "eng")):
            text = _ocr(img, psm, lang)
            if not text:
                continue
            parsed = parse_nutrients(text)
            score = (len(parsed) * 100) + _keyword_hit_count(text)
            texts.append((text, score))
            parses.append(parsed)

    if not texts:
        return "", {}

    voted: dict[str, float] = {}
    all_keys = set().union(*(p.keys() for p in parses))
    for key in all_keys:
        values = [p[key] for p in parses if key in p]
        counts = Counter(values).most_common()
        if counts:
            voted[key] = counts[0][0]

    raw_text = max(texts, key=lambda item: item[1])[0]
    return raw_text, voted


def process_ocr_image(image_bytes: bytes) -> dict:
    if not image_bytes:
        return {
            "status": "error",
            "engine": "none",
            "nutrients": {},
            "raw_text": "",
            "message": "Empty image bytes",
        }

    raw_text: str | None = None
    nutrients: dict[str, float] = {}
    engine = "google"

    g_text = _run_google_vision(image_bytes)
    g_nutrients: dict[str, float] = {}
    if g_text:
        g_nutrients = parse_nutrients(g_text)

    # Google Vision is always preferred for Bangla text — its output is far more
    # accurate than Tesseract. Only fall back to Tesseract when Vision returns
    # nothing at all (credentials missing, network error, etc.).
    if g_text and g_text.strip():
        raw_text = g_text
        nutrients = g_nutrients
        engine = "google"
    else:
        try:
            tess_text, tess_nutrients = _run_tesseract_multi(image_bytes)
        except ValueError as exc:
            return {
                "status": "error",
                "engine": "none",
                "nutrients": {},
                "raw_text": "",
                "message": str(exc),
            }

        raw_text = tess_text or ""
        nutrients = tess_nutrients
        engine = "tesseract" if tess_text else "none"

    if not raw_text or not raw_text.strip():
        return {"status": "no_text", "engine": engine, "nutrients": {}, "raw_text": ""}

    return {
        "status": "success" if nutrients else "no_text",
        "engine": engine,
        "nutrients": nutrients,
        "raw_text": raw_text,
    }


if __name__ == "__main__":
    sample = """
    Nutrition Facts
    Serving size 30g
    Calories 230
    Protein 8 g
    Total Fat 12g
    Carbohydrate 28g
    Sugars 10 g
    Sodium 320 mg
    Cholesterol 5 mg
    Dietary Fiber 4g
    """
    print(parse_nutrients(sample))
