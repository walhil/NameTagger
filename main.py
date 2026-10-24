from src import vision
from src.bot import ROSTER_REFRESH_HOURS, bot
from src.config import DISCORD_TOKEN, GSHEETS_SHEET_NAME, MEMBER_ROLE, OCR_LANGS


def print_startup_config() -> None:
    print("=== NameTagger starting ===")
    print(f"  Vision model:   {vision.MODEL} (effort={vision.EFFORT}, max_tokens={vision.MAX_TOKENS})")
    print(f"  OCR languages:  {', '.join(OCR_LANGS)}")
    print(f"  Member role:    {MEMBER_ROLE}")
    print(f"  Roster sheet:   {GSHEETS_SHEET_NAME} (refresh every {ROSTER_REFRESH_HOURS}h)")
    print("===========================")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    print_startup_config()
    bot.run(DISCORD_TOKEN)
