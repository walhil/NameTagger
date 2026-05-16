# NameTagger

A Discord bot for guild loot sessions that uses OCR to read player names from screenshots, match them to Discord members, and extract loot/deposit values for split calculations.

## Workflow

1. Upload a screenshot of the party.
2. Run `!ping` — the bot scans recent images, reads player names via OCR, matches them to Discord members, and @mentions each one.
3. Upload screenshots of the loot inventory and guild silver deposits.
4. Run `!scan` — the bot reads the Est. Market Value from the loot screenshot and sums all deposit amounts from the deposit log screenshots. Outputs pre-filled arguments for `/split-loot`.

## Commands

| Command | Description |
|---|---|
| `!ping` | Scans the last 50 messages for party screenshots, OCRs player names, and @mentions matched Discord members. |
| `!scan` | Scans images uploaded after the last `!ping` for loot and deposit screenshots. Outputs the total loot value, total deposited silver, and a ready-to-use `/split-loot` command. |

## Setup

### Requirements

```
discord.py
easyocr
opencv-python
python-dotenv
google-api-python-client
google-auth
difflib
```

### Environment Variables

Create a `.env` file in the project root:

```
DISCORD_TOKEN=your_discord_bot_token
GSHEETS_SPREADSHEET_ID=your_google_sheet_id
```

### Google Sheets

The bot loads the player roster from a Google Sheet on startup.

- Sheet name: `Data Validation`
- Column header: `Player IGM`

Place a valid `token.json` (OAuth2 credentials) in the same directory as `NameDetect.py`. The bot will automatically refresh the token when needed.

### Running

```bash
python NameDetect.py
```

### Deployment

The bot is designed to run as a `systemd` service. Example unit file:

```ini
[Unit]
Description=DiscordNameTaggerBot

[Service]
WorkingDirectory=/home/ubuntu/discord-bot/NameTagger
ExecStart=/home/ubuntu/discord-bot/NameTagger/venv/bin/python NameDetect.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
