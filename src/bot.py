import asyncio
import re

import aiohttp
import discord
from discord.ext import commands, tasks

from .config import DISCORD_TOKEN, EVENT_TYPES, MEMBER_ROLE, SPLIT_TYPES
from .ocr import extract_names
from .roster import correct_name, find_member, load_roster
from .vision import scan_image

ROSTER_REFRESH_HOURS = 3

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# channel id -> (!ping message id, [(attachment, message)]): loot/deposit images that
# !ping skipped, handed to the next !scan if it runs against that same !ping.
_carried_to_scan: dict[int, tuple[int, list[tuple]]] = {}


async def _download(attachment, retries: int = 3, delay: float = 1.0) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            return await attachment.read()
        except (aiohttp.ClientPayloadError, aiohttp.http_exceptions.ContentLengthError) as e:
            if attempt == retries:
                raise
            print(f"[download] Attempt {attempt} failed: {e}")
            await asyncio.sleep(delay)


def _is_image(attachment) -> bool:
    return (
        attachment.content_type and attachment.content_type.startswith("image")
    ) or attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))


@tasks.loop(hours=ROSTER_REFRESH_HOURS)
async def _refresh_roster_loop():
    load_roster()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    if not _refresh_roster_loop.is_running():
        _refresh_roster_loop.start()


@bot.command()
async def refreshroster(ctx: commands.Context):
    """Reload the player roster from Google Sheets immediately."""
    load_roster()
    await ctx.send("Roster reloaded from Google Sheets.")


@bot.command()
async def ping(ctx: commands.Context):
    """Scan recent images for player names and @mention matched Discord members."""
    MESSAGE_LIMIT = 50

    await ctx.send(f"Scanning the last {MESSAGE_LIMIT} messages for party images...")

    # (attachment, message, is_current_run). Images older than the last !scan belong
    # to a previous split, so they are never carried over to this run's !scan.
    image_entries = []
    is_current_run = True
    async for msg in ctx.channel.history(limit=MESSAGE_LIMIT):
        if msg.id != ctx.message.id and msg.content.strip().lower().startswith("!scan"):
            is_current_run = False
        for att in msg.attachments:
            if _is_image(att):
                image_entries.append((att, msg, is_current_run))

    if not image_entries:
        await ctx.send("No image attachments found in recent messages.")
        return

    await ctx.send(f"Found **{len(image_entries)}** image(s). Running OCR...")

    loop = asyncio.get_event_loop()
    all_names: list[str] = []
    skipped = 0
    carried: list[tuple] = []
    for att, msg, current in image_entries:
        try:
            image_bytes = await _download(att)
            names, skipped_kind = await loop.run_in_executor(None, extract_names, image_bytes)
            if skipped_kind:
                skipped += 1
                if current:
                    carried.append((att, msg))
            all_names.extend(names)
        except Exception as e:
            print(f"[ping] Error on {att.filename}: {e}")

    _carried_to_scan[ctx.channel.id] = (ctx.message.id, carried)

    if carried:
        await ctx.send(
            f"Skipped **{len(carried)}** loot/deposit image(s) posted before `!ping`. "
            f"They'll be included in `!scan`, so there's no need to re-post them."
        )

    if not all_names:
        await ctx.send("Could not detect any names from the images.")
        return

    seen = set()
    unique_names = []
    for n in all_names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique_names.append(n)

    member_hits: dict = {}
    unmatched: list[tuple[str, str]] = []

    for n in unique_names:
        corrected = correct_name(n)
        if corrected is None:
            continue
        member = await find_member(ctx.guild, corrected)
        if member and any(r.name == MEMBER_ROLE for r in member.roles):
            entry = member_hits.setdefault(
                member.id, {"member": member, "raw": set(), "corrected": set()}
            )
            entry["raw"].add(n)
            entry["corrected"].add(corrected)
        else:
            unmatched.append((n, corrected))

    matched_lines = []
    for entry in member_hits.values():
        member = entry["member"]
        raw_names = sorted(entry["raw"])
        corrected_names = sorted(entry["corrected"])
        if len(corrected_names) == 1:
            corrected = corrected_names[0]
            raw_unique = [r for r in raw_names if r.lower() != corrected.lower()]
            if raw_unique:
                line = f"{', '.join(f'`{r}`' for r in raw_unique)}→`{corrected}`→{member.mention}"
            else:
                line = f"`{corrected}`→{member.mention}"
        else:
            line = (
                f"{', '.join(f'`{r}`' for r in raw_names)}"
                f"→{', '.join(f'`{c}`' for c in corrected_names)}"
                f"→{member.mention}"
            )
        matched_lines.append(line)

    unmatched_lines = [
        f"`{raw}`→`{corr}` (no match)" if corr != raw else f"`{raw}` (no match)"
        for raw, corr in unmatched
    ]

    MAX_DISPLAY = 60
    display_matched = matched_lines[:MAX_DISPLAY]
    display_unmatched = unmatched_lines[:max(0, MAX_DISPLAY - len(display_matched))]
    extra_matched = len(matched_lines) - len(display_matched)
    extra_unmatched = len(unmatched_lines) - len(display_unmatched)

    parts = [f"**Matched {len(member_hits)} player(s) from {len(image_entries) - skipped} image(s).**"]
    if display_matched:
        parts += ["\n**Matched:**"] + display_matched
    if display_unmatched:
        parts += ["\n**Unmatched:**"] + display_unmatched
    if extra_matched or extra_unmatched:
        parts.append(f"\n…and {extra_matched} more matched, {extra_unmatched} more unmatched not shown.")

    text = "\n".join(parts)
    if len(text) > 2000:
        text = text[:1990] + "\n…(truncated)"
    await ctx.send(text)


class _CallerSelect(discord.ui.UserSelect):
    def __init__(self):
        super().__init__(placeholder="Select caller...", row=2)

    async def callback(self, interaction: discord.Interaction):
        self.view.caller = self.values[0]
        await interaction.response.defer()


class OverrideValuesModal(discord.ui.Modal, title="Override Values"):
    loot_input = discord.ui.TextInput(
        label="Loot Value (silver)",
        placeholder="e.g. 8180000",
        required=False,
    )
    deposit_input = discord.ui.TextInput(
        label="Total Deposited (silver)",
        placeholder="e.g. 3920000",
        required=False,
    )

    def __init__(self, view: "ScanConfigView"):
        super().__init__()
        self._view = view
        if view.market_value is not None:
            self.loot_input.default = str(view.market_value)
        if view.silver_total:
            self.deposit_input.default = str(view.silver_total)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            raw = self.loot_input.value.strip().replace(",", "")
            if raw:
                self._view.market_value = int(raw)
        except ValueError:
            pass
        try:
            raw = self.deposit_input.value.strip().replace(",", "")
            if raw:
                self._view.silver_total = int(raw)
        except ValueError:
            pass
        self._view._refresh_display()
        await interaction.response.edit_message(content=self._view._build_status(), view=self._view)


class ScanConfigView(discord.ui.View):
    def __init__(self, market_value: int | None, silver_total: int):
        super().__init__(timeout=180)
        self.split_type: str | None = None
        self.event_type: str | None = None
        self.caller: discord.Member | None = None
        self.is_damaged: bool = False
        self._done = asyncio.Event()
        self.market_value = market_value
        self.silver_total = silver_total
        self._loot_str = f"{market_value:,}" if market_value is not None else "not found"
        self._silver_str = f"{silver_total:,}" if silver_total else "not found"
        self.add_item(_CallerSelect())

    def _refresh_display(self) -> None:
        self._loot_str = f"{self.market_value:,}" if self.market_value is not None else "not found"
        self._silver_str = f"{self.silver_total:,}" if self.silver_total else "not found"

    def _build_status(self) -> str:
        return (
            f"**Est. Market Value:** {self._loot_str} silver\n"
            f"**Total Deposited:** {self._silver_str} silver\n\n"
            f"**Split Type:** {self.split_type or '—'}\n"
            f"**Event Type:** {self.event_type or '—'}\n"
            f"**Caller:** {self.caller.display_name if self.caller else '—'}\n"
            f"**Loot:** {'Damaged' if self.is_damaged else 'Not Damaged'}"
        )

    @discord.ui.select(
        placeholder="Select split type...",
        options=[discord.SelectOption(label=t) for t in SPLIT_TYPES],
        row=0,
    )
    async def select_split_type(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.split_type = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="Select event type...",
        options=[discord.SelectOption(label=t) for t in EVENT_TYPES],
        row=1,
    )
    async def select_event_type(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.event_type = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="Select loot condition...",
        options=[
            discord.SelectOption(label="Not Damaged", value="not_damaged"),
            discord.SelectOption(label="Damaged", value="damaged"),
        ],
        row=3,
    )
    async def select_damaged(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.is_damaged = select.values[0] == "damaged"
        await interaction.response.defer()

    @discord.ui.button(label="Edit Silver Values", style=discord.ButtonStyle.grey, row=4)
    async def edit_values(self, interaction: discord.Interaction, _button: discord.ui.Button):
        await interaction.response.send_modal(OverrideValuesModal(self))

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.blurple, row=4)
    async def confirm(self, interaction: discord.Interaction, _button: discord.ui.Button):
        missing = [
            name for name, val in [
                ("split type", self.split_type),
                ("event type", self.event_type),
                ("caller", self.caller),
            ] if not val
        ]
        if missing:
            await interaction.response.send_message(
                f"Please select: {', '.join(missing)}.", ephemeral=True
            )
            return
        self.clear_items()
        await interaction.response.edit_message(content="✓ Confirmed.", view=self)
        self._done.set()


@bot.command()
async def scan(ctx: commands.Context):
    """Scan loot and deposit images posted after the last !ping, then collect split details."""
    MESSAGE_LIMIT = 100

    image_entries = []
    ping_msg_id = None
    async for msg in ctx.channel.history(limit=MESSAGE_LIMIT):
        if msg.content.strip().lower().startswith("!ping"):
            ping_msg_id = msg.id
            break
        for att in msg.attachments:
            if _is_image(att):
                image_entries.append((att, msg))

    # Loot/deposit images posted before !ping (which !ping skipped). Appended last so
    # the order stays newest-first, matching the history walk above.
    carried_ping_id, carried = _carried_to_scan.get(ctx.channel.id, (None, []))
    if carried and carried_ping_id == ping_msg_id:
        image_entries.extend(carried)
    else:
        carried = []

    if not image_entries:
        await ctx.send("No images found after the last `!ping`.")
        return

    found = f"Found **{len(image_entries)}** image(s)"
    if carried:
        found += f" ({len(carried)} posted before `!ping`)"
    await ctx.send(f"{found}. Analysing...")

    market_value: int | None = None
    deposit_amounts: list[int] = []

    for att, _ in image_entries:
        try:
            image_bytes = await _download(att)
            mv, deps = await scan_image(image_bytes, att.filename)
        except Exception as e:
            print(f"[scan] Error on {att.filename}: {e}")
            continue

        if mv is not None:
            market_value = mv
        deposit_amounts.extend(deps)

    silver_total = sum(deposit_amounts) if deposit_amounts else 0

    view = ScanConfigView(market_value, silver_total)
    await ctx.send(view._build_status(), view=view)

    try:
        await asyncio.wait_for(view._done.wait(), timeout=180)
    except asyncio.TimeoutError:
        await ctx.send("Timed out. Run `!scan` again.")
        return

    loot_val = view.market_value or 0
    silver_val = view.silver_total
    loot_key = "damaged-loot-total" if view.is_damaged else "non-damaged-loot-total"
    loot_label = "Damaged loot" if view.is_damaged else "Non-damaged loot"

    caller_name = re.sub(r"^!+sl\s*", "", view.caller.display_name, flags=re.IGNORECASE).strip()

    summary = (
        f"⚔️ **Split Summary**\n\n"
        f"> 👤 **Caller:** {view.caller.mention}\n"
        f"> 🏰 **Event:** {view.split_type} · {view.event_type}\n"
        f"> 🗡️ **Loot:** {loot_label} · {loot_val:,} silver\n"
        f"> 💰 **Deposits:** {silver_val:,} silver\n\n"
        f"▶ `/split-loot loot-split-type:{view.split_type} caller-name:@{caller_name} "
        f"event-type:{view.event_type} {loot_key}:{loot_val} silver-bags-total:{silver_val}`"
    )
    await ctx.send(summary)
