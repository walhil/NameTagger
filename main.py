from src.bot import bot
from src.config import DISCORD_TOKEN

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    bot.run(DISCORD_TOKEN)
