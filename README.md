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
- **Damaged / Not Damaged** dropdown
- **Confirm** button

Every selection updates the message live so all thread members can see the current state.

## File Structure

```
main.py               # Entry point
src/
  bot.py              # Bot setup and commands
  config.py           # Constants and environment variables
  ocr.py              # Image preprocessing, OCR, name/value extraction
  roster.py           # Google Sheets roster loading and member matching
  vision.py           # Claude vision API for loot/deposit value extraction
requirements.txt
Dockerfile            # Container image (CPU-only PyTorch, EasyOCR models built in)
docker-compose.yml    # Runs the bot; mounts ./data for token.json
README.md
```

### Files not included in the repository

These files are required at runtime but excluded from version control via `.gitignore`:

| File | Description |
|---|---|
| `.env` | Environment variables (see below) |
| `token.json` | Google OAuth2 token for Sheets API access |
| `client_secret.json` | Google OAuth2 client credentials |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root with the following:

```env
DISCORD_TOKEN=your_discord_bot_token
ANTHROPIC_API_KEY=your_anthropic_api_key
GSHEETS_SPREADSHEET_ID=your_google_sheet_id
```

| Variable | Where to get it |
|---|---|
| `DISCORD_TOKEN` | [Discord Developer Portal](https://discord.com/developers/applications) → Your App → Bot → Token |
| `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) → API Keys |
| `GSHEETS_SPREADSHEET_ID` | Your Google Sheet URL: `https://docs.google.com/spreadsheets/d/<ID>/edit` |

### 3. Google Sheets OAuth

The bot reads the player roster from a Google Sheet on startup.

- Sheet name: `Data Validation`
- Column header: `Player IGM`

To authenticate:
1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Google Sheets API
3. Create OAuth 2.0 credentials (Desktop app) and download as `client_secret.json`
4. Run the OAuth flow once to generate `token.json`
5. Place both files in the project root alongside `main.py`

The bot will automatically refresh `token.json` when the access token expires.

### 4. Run

```bash
python main.py
```

## Deployment

### Docker Compose (recommended)

Requires Docker with the Compose plugin.

```bash
git clone https://github.com/walhil/NameTagger.git && cd NameTagger

# 1. Create .env (see "Environment variables" above)
nano .env

# 2. Put token.json in ./data (the container runs as UID 1000 and must be able to write it)
mkdir -p data
cp /path/to/token.json data/
sudo chown -R 1000:1000 data

# 3. Build and start
docker compose up -d --build
```

Useful commands:

| Command | Description |
|---|---|
| `docker compose logs -f` | Follow bot logs |
| `docker compose restart` | Restart the bot |
| `git pull && docker compose up -d --build` | Update to the latest code |
| `docker compose down` | Stop and remove the container |

Notes:
- The image uses CPU-only PyTorch and has the EasyOCR models built in. The first build takes several minutes, and the image is about 3 GB.
- The container needs ~1 GB of RAM at runtime (see the swap note below for small VPSes).
- `.env`, `token.json`, and `*.pem` are excluded from the image via `.dockerignore`.

### systemd

The bot can also run as a `systemd` service on Linux.

Example unit file (`/etc/systemd/system/discord-bot.service`):

```ini
[Unit]
Description=DiscordNameTaggerBot
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/discord-bot/NameTagger
ExecStart=/home/ubuntu/discord-bot/NameTagger/venv/bin/python main.py
Restart=on-failure
User=ubuntu

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable discord-bot.service
sudo systemctl start discord-bot.service
```

> **Note:** EasyOCR requires ~1 GB of RAM at runtime. If deploying on a low-memory instance, add at least 2 GB of swap:
> ```bash
> sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
> sudo mkswap /swapfile && sudo swapon /swapfile
> echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
> ```
