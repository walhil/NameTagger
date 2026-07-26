import difflib
import os
import re

import discord
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from .config import (
    GSHEETS_COLUMN_HEADER,
    GSHEETS_SCOPES,
    GSHEETS_SHEET_NAME,
    GSHEETS_SPREADSHEET_ID,
    GSHEETS_TOKEN_FILE,
)

_roster_names: list[str] = []
_roster_norm_map: dict[str, str] = {}

_OCR_DENYLIST: frozenset[str] = frozenset({
    "name", "search", "tier", "category", "actions", "equip", "sort",
    "repair", "move", "all", "item", "items", "select", "loadout",
    "weight", "durability", "quality", "value", "type", "slot",
})


def _normalize(s: str) -> str:
    s = re.sub(r"^[^A-Za-z]+", "", s.strip())
    s = re.sub(r"\s+", "", s).lower()
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def _get_creds() -> Credentials:
    creds = None
    if os.path.exists(GSHEETS_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GSHEETS_TOKEN_FILE, GSHEETS_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[ROSTER] Refreshing Google Sheets access token...")
            creds.refresh(Request())
            with open(GSHEETS_TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        else:
            raise RuntimeError(
                "[ROSTER] token.json is missing or invalid. "
                "It must contain token, refresh_token, client_id, client_secret, token_uri, and scopes."
            )
    return creds


def load_roster() -> None:
    global _roster_names, _roster_norm_map

    if not GSHEETS_SPREADSHEET_ID:
        print("[ROSTER] GSHEETS_SPREADSHEET_ID not set. Skipping roster load.")
        return

    try:
        creds = _get_creds()
        service = build("sheets", "v4", credentials=creds)
        result = (
            service.spreadsheets()
            .values()
            .get(
                spreadsheetId=GSHEETS_SPREADSHEET_ID,
                range=f"{GSHEETS_SHEET_NAME}!A:Z",
            )
            .execute()
        )
        values = result.get("values", [])
    except Exception as e:
        print(f"[ROSTER] Failed to load roster: {e}")
        return

    if not values:
        print("[ROSTER] No data found in Google Sheet.")
        return

    try:
        col_index = values[0].index(GSHEETS_COLUMN_HEADER)
    except ValueError:
        print(f"[ROSTER] Column '{GSHEETS_COLUMN_HEADER}' not found.")
        return

    names = [
        str(row[col_index]).strip()
        for row in values[1:]
        if len(row) > col_index and row[col_index]
    ]

    _roster_names = names
    _roster_norm_map = {_normalize(n): n for n in _roster_names}
    print(f"[ROSTER] Loaded {len(_roster_names)} names from Google Sheets.")


def correct_name(ocr_name: str, cutoff: float = 0.6) -> str | None:
    if not _roster_norm_map:
        return ocr_name
    target = _normalize(ocr_name)
    if not target or target in _OCR_DENYLIST:
        return None
    best = difflib.get_close_matches(target, list(_roster_norm_map), n=1, cutoff=cutoff)
    if best and len(target) >= len(best[0]) * 0.5:
        corrected = _roster_norm_map[best[0]]
        print(f"[ROSTER] {ocr_name!r} -> {corrected!r}")
        return corrected
    return ocr_name


async def find_member(guild: discord.Guild, name: str) -> discord.Member | None:
    if not guild:
        return None
    target = _normalize(name)
    if not target:
        return None

    members = [m async for m in guild.fetch_members(limit=None)]

    for m in members:
        if target in (_normalize(m.name), _normalize(m.display_name)):
            return m
    for m in members:
        if target in _normalize(m.name) or target in _normalize(m.display_name):
            return m
    return None
