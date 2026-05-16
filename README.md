# NameTagger

A Discord bot for guild loot sessions that uses OCR to read player names from screenshots, match them to Discord members, and extract loot/deposit values for split calculations.

## Workflow

1. Upload a screenshot of the party members.
2. Run `!ping` — the bot OCRs player names, matches them against the guild roster and Discord members, and @mentions each one.
3. Upload screenshots of the loot inventory and guild silver deposits.
4. Run `!scan` — the bot reads the Est. Market Value and sums all deposits, then presents an interactive form to configure the split.
5. Hit **Confirm** — the bot outputs a pre-filled `/split-loot` command ready to run.

## Commands

| Command | Description |
|---|---|
| `!ping` | Scans the last 50 messages for party screenshots, OCRs player names, and @mentions matched Discord members. |
| `!scan` | Scans images uploaded after the last `!ping` for loot and deposit screenshots. Presents a form to select split type, event type, caller, and loot condition. Outputs a ready-to-use `/split-loot` command. |

## `!scan` Form

After OCR completes, the bot posts an interactive message with:
- **Split Type** dropdown (Guild, Personal, OffSeason, Other)
- **Event Type** dropdown (Bandit, Castle, Chest, Ganking, Hellgates, etc.)
- **Caller** user selector
- **Damaged / Not Damaged** toggle
- **Confirm** button

Every selection updates the message live so all thread members can see the current state.

## File Structure

```
NameDetect.py   # Entry point — bot setup and commands
config.py       # Constants and environment variables
ocr.py          # Image preprocessing, OCR, name/value extraction
roster.py       # Google Sheets roster loading and member matching
requirements.txt
```

## Setup

### Requirements

```
pip install -r requirements.txt
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
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/discord-bot/NameTagger
ExecStart=/home/ubuntu/discord-bot/NameTagger/venv/bin/python NameDetect.py
Restart=on-failure
User=ubuntu

[Install]
WantedBy=multi-user.target
```

> **Note:** EasyOCR requires ~1 GB of RAM at runtime. If deploying on a low-memory instance, add at least 2 GB of swap.
