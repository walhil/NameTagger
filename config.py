import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OCR
OCR_LANGS = ["en"]
OCR_MIN_CONF = 0.2
OCR_SCALE = 2.0

# Loot split options
SPLIT_TYPES = ["Guild", "Personal", "OffSeason", "Other"]
EVENT_TYPES = [
    "Bandit", "Castle", "Chest", "Ganking", "Hellgates", "Hideout",
    "Mages", "Orb", "Outpost", "Other", "SpecialEvent", "Territory",
    "Vortex", "DrinkingBeer",
]

# Google Sheets
GSHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
GSHEETS_SPREADSHEET_ID = os.getenv("GSHEETS_SPREADSHEET_ID")
GSHEETS_SHEET_NAME = "Data Validation"
GSHEETS_COLUMN_HEADER = "Player IGM"
GSHEETS_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
