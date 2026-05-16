import re

import cv2
import easyocr
import numpy as np

from config import OCR_LANGS, OCR_MIN_CONF, OCR_SCALE

print("Initializing EasyOCR reader...")
_reader = easyocr.Reader(OCR_LANGS)
print("EasyOCR reader ready.")


def _preprocess(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Failed to decode image bytes with OpenCV.")
    img = cv2.resize(img_bgr, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run_ocr(image_bytes: bytes):
    return _reader.readtext(_preprocess(image_bytes), detail=1)


# ---- Name extraction ----

def _is_probable_name(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return False
    if not any(c.isalpha() for c in text):
        return False
    if "/" in text or text.count(" ") > 1:
        return False
    return True


def _normalize_name(text: str) -> str:
    n = text.strip().strip("[]{}()<>|\"'`.,;:!?")
    n = re.sub(r"^[^A-Za-z0-9]+", "", n)
    if not n:
        return n
    n = re.sub(r"\s+", "", n)
    if len(n) >= 2 and n[0].isdigit() and any(c.isalpha() for c in n[1:]):
        n = n[1:]
    n = re.sub(r"(.)\1{2,}", r"\1\1", n)
    return n[0].upper() + n[1:]


def extract_names(image_bytes: bytes) -> list[str]:
    results = run_ocr(image_bytes)

    print("Raw OCR results:")
    for i, (_, text, conf) in enumerate(results, start=1):
        print(f"  {i}. [{conf:.2f}] {text}")

    names = []
    for _, text, conf in results:
        if conf < OCR_MIN_CONF or not _is_probable_name(text):
            continue
        cleaned = _normalize_name(text)
        if cleaned:
            names.append(cleaned)

    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique.append(n)
    return unique


# ---- Loot / deposit extraction ----

def parse_silver_amount(text: str) -> int | None:
    text = text.strip().replace(",", "").replace(" ", "")
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)[mM]$", text)
    if m:
        return int(round(float(m.group(1)) * 1_000_000))
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)[kK]$", text)
    if m:
        return int(round(float(m.group(1)) * 1_000))
    m = re.match(r"^([0-9]+)$", text)
    if m:
        return int(m.group(1))
    return None


def extract_market_value(ocr_results) -> int | None:
    full_text = " ".join(text for _, text, _ in ocr_results)
    m = re.search(
        r"[Ee]st[.;,]?\s*[Mm]arke[tr]\s*[Vv]alue\s*[:\s]+[^0-9]*([0-9][0-9,.]*\s*[mkMK]?)",
        full_text,
    )
    if m:
        return parse_silver_amount(m.group(1))
    return None


def extract_deposit_amounts(ocr_results) -> list[int]:
    full_text = " ".join(text for _, text, _ in ocr_results)
    amounts = []
    for m in re.finditer(r"[Dd]epos[a-zA-Z]+\s+([0-9][0-9,.]*)", full_text):
        amount = parse_silver_amount(m.group(1))
        if amount is not None:
            amounts.append(amount)
    return amounts
