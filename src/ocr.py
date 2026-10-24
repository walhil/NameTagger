import difflib
import re

import cv2
import easyocr
import numpy as np

from .config import OCR_LANGS, OCR_MIN_CONF, OCR_SCALE

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


def _preprocess_scan(image_bytes: bytes) -> np.ndarray:
    """3x scale + binary threshold — better for reading numbers in deposit logs."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Failed to decode image bytes with OpenCV.")
    img = cv2.resize(img_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary


def run_ocr_scan(image_bytes: bytes):
    """Run OCR with preprocessing tuned for loot and deposit screenshots."""
    return _reader.readtext(_preprocess_scan(image_bytes), detail=1)


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


# UI labels from the bank/inventory screen. OCR mangles these a lot ("Move Aii",
# "RepalrAI", "Tler"), so each OCR box is fuzzy-matched against them, and several
# distinct labels must hit so a player who happens to be named "Stack" can't flag a party list.
_LOOT_MARKERS = (
    "marketvalue", "moveall", "repairall", "selectloadout", "equip",
    "search", "tier", "category", "sort", "stack", "bank", "actions",
)
_LOOT_MIN_HITS = 3
_DEPOSIT_MARKERS = ("deposit", "withdrawal")


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fuzzy_hit(box: str, marker: str) -> bool:
    return marker in box or difflib.SequenceMatcher(None, box, marker).ratio() >= 0.75


def detect_non_party(ocr_results) -> str | None:
    """Return 'loot' or 'deposit' if the OCR text looks like one of those screens, else None."""
    boxes = [_squash(t) for _, t, _ in ocr_results]
    boxes = [b for b in boxes if b]
    if any(_fuzzy_hit(b, m) for b in boxes for m in _DEPOSIT_MARKERS):
        return "deposit"
    hits = sum(any(_fuzzy_hit(b, m) for b in boxes) for m in _LOOT_MARKERS)
    if hits >= _LOOT_MIN_HITS:
        return "loot"
    return None


def extract_names(image_bytes: bytes) -> tuple[list[str], str | None]:
    """
    Return (names, skipped_kind). skipped_kind is 'loot' or 'deposit' when the image
    is not a party list, in which case names is empty.
    """
    results = run_ocr(image_bytes)

    print("Raw OCR results:")
    for i, (_, text, conf) in enumerate(results, start=1):
        print(f"  {i}. [{conf:.2f}] {text}")

    kind = detect_non_party(results)
    if kind:
        print(f"Skipping image: looks like a {kind} screenshot, not a party list.")
        return [], kind

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
    return unique, None


# ---- Loot / deposit extraction ----

def parse_silver_amount(text: str) -> int | None:
    text = text.strip().replace(",", "").replace(" ", "")
    # OCR sometimes reads lowercase 'm' (million) as 'In' or 'in'
    if text.lower().endswith("in"):
        text = text[:-2] + "m"
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
        r"[Ee]s[tc][.;,_]?\s*[Mm]arke[tr]\s*[Vv][ao][ll][uo][eu][:\s]+[^0-9]*([0-9][0-9,.]*\s*(?:[mkMK]|[Ii]n)?)",
        full_text,
    )
    if m:
        return parse_silver_amount(m.group(1))
    return None


def extract_deposit_amounts(ocr_results) -> list[int]:
    full_text = " ".join(text for _, text, _ in ocr_results)
    amounts = []
    for deposit_m in re.finditer(r"[Dd][Ee]?[Pp][Oo][Ss][Ii][Cc]?[Tt]?", full_text):
        segment = full_text[deposit_m.end():]
        next_dep = re.search(r"[Dd][Ee]?[Pp][Oo][Ss][Ii][Cc]?[Tt]?", segment)
        if next_dep:
            segment = segment[:next_dep.start()]
        candidates = re.findall(r"[0-9]{1,3}(?:,[0-9]{3})+", segment)
        if candidates:
            parsed = [parse_silver_amount(n) for n in candidates]
            parsed = [v for v in parsed if v is not None]
            if parsed:
                amounts.append(max(parsed))
    return amounts
