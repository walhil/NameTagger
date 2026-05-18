import base64
import re

import anthropic

from .config import ANTHROPIC_API_KEY

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-haiku-4-5-20251001"


def _media_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")


async def _invoke(prompt: str, image_bytes: bytes, filename: str) -> str:
    msg = await _client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _media_type(filename),
                            "data": base64.standard_b64encode(image_bytes).decode(),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return msg.content[0].text.strip()


async def scan_image(image_bytes: bytes, filename: str) -> tuple[int | None, list[int]]:
    """
    Send one image to Claude and extract whichever values are present:
    - Est. Market Value (from a loot inventory screenshot)
    - Deposit amounts (from a guild deposit log screenshot)
    Returns (market_value, deposit_amounts).
    """
    prompt = (
        "Analyze this screenshot. It will be one of two things:\n\n"
        "1. A game inventory or bank screen. At the very bottom you will see a label that says "
        "'Est. Market Value:' followed by a number and a unit suffix. "
        "The suffix 'm' means millions and 'k' means thousands — convert accordingly "
        "(e.g. '8.18 m' = 8180000, '4.5 m' = 4500000, '18.7 m' = 18700000). "
        "Read ONLY the number next to that specific label. Ignore all other numbers in the image.\n\n"
        "2. A guild deposit log with columns for date, player name, action type, and silver amount.\n\n"
        "If it is an inventory/bank screen, respond with exactly:\n"
        "MARKET: <integer> (e.g. MARKET: 8180000)\n\n"
        "If it is a deposit log, respond with exactly:\n"
        "DEPOSITS: <comma-separated integers> (e.g. DEPOSITS: 1190000,1500000,1230000)\n"
        "Only include rows where the action column says 'Deposit'. Ignore 'Withdrawal' rows.\n\n"
        "If neither applies: NONE"
    )

    try:
        result = await _invoke(prompt, image_bytes, filename)
        print(f"[VISION] {filename}: {repr(result)}")
    except Exception as e:
        print(f"[VISION] Error on {filename}: {e}")
        return None, []

    market_value = None
    deposit_amounts = []

    if result.startswith("MARKET:"):
        raw = result.removeprefix("MARKET:").strip().replace(",", "")
        m = re.match(r"^([0-9]+)$", raw)
        if m:
            market_value = int(m.group(1))

    elif result.startswith("DEPOSITS:"):
        raw = result.removeprefix("DEPOSITS:").strip()
        for token in raw.split(","):
            token = token.strip().replace(",", "")
            m = re.match(r"^([0-9]+)$", token)
            if m:
                deposit_amounts.append(int(m.group(1)))

    return market_value, deposit_amounts
