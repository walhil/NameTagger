import asyncio

import aiohttp
import discord
from discord.ext import commands

from config import DISCORD_TOKEN, EVENT_TYPES, SPLIT_TYPES
from ocr import extract_deposit_amounts, extract_market_value, extract_names, run_ocr
from roster import correct_name, find_member, load_roster

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


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


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    load_roster()


@bot.command()
async def ping(ctx: commands.Context):
    """Scan recent images for player names and @mention matched Discord members."""
    MESSAGE_LIMIT = 50

    await ctx.send(f"Scanning the last {MESSAGE_LIMIT} messages for party images...")

    image_entries = [
        att
        async for msg in ctx.channel.history(limit=MESSAGE_LIMIT)
        for att in msg.attachments
        if _is_image(att)
    ]

    if not image_entries:
        await ctx.send("No image attachments found in recent messages.")
        return

    await ctx.send(f"Found **{len(image_entries)}** image(s). Running OCR...")

    all_names: list[str] = []
    for att in image_entries:
        try:
            names = extract_names(await _download(att))
            all_names.extend(names)
        except Exception as e:
            print(f"[ping] Error on {att.filename}: {e}")

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
        member = await find_member(ctx.guild, corrected)
        if member:
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

    parts = [f"**Matched {len(member_hits)} player(s) from {len(image_entries)} image(s).**"]
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
        await interaction.response.edit_message(content=self.view._build_status(), view=self.view)


class ScanConfigView(discord.ui.View):
    def __init__(self, loot_str: str, silver_str: str):
        super().__init__(timeout=180)
        self.split_type: str | None = None
        self.event_type: str | None = None
        self.caller: discord.Member | None = None
        self.is_damaged: bool = False
        self._done = asyncio.Event()
        self._loot_str = loot_str
        self._silver_str = silver_str
        self.add_item(_CallerSelect())

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
        await interaction.response.edit_message(content=self._build_status(), view=self)

    @discord.ui.select(
        placeholder="Select event type...",
        options=[discord.SelectOption(label=t) for t in EVENT_TYPES],
        row=1,
    )
    async def select_event_type(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.event_type = select.values[0]
        await interaction.response.edit_message(content=self._build_status(), view=self)

    @discord.ui.button(label="Not Damaged", style=discord.ButtonStyle.green, row=3)
    async def toggle_damaged(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.is_damaged = not self.is_damaged
        button.label = "Damaged" if self.is_damaged else "Not Damaged"
        button.style = discord.ButtonStyle.red if self.is_damaged else discord.ButtonStyle.green
        await interaction.response.edit_message(content=self._build_status(), view=self)

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
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content=self._build_status(), view=self)
        self._done.set()


@bot.command()
async def scan(ctx: commands.Context):
    """Scan loot and deposit images posted after the last !ping, then collect split details."""
    MESSAGE_LIMIT = 100

    image_entries = []
    async for msg in ctx.channel.history(limit=MESSAGE_LIMIT):
        if msg.content.strip().lower().startswith("!ping"):
            break
        for att in msg.attachments:
            if _is_image(att):
                image_entries.append((att, msg))

    if not image_entries:
        await ctx.send("No images found after the last `!ping`.")
        return

    await ctx.send(f"Found **{len(image_entries)}** image(s) since last `!ping`. Running OCR...")

    market_value: int | None = None
    deposit_amounts: list[int] = []

    for att, _ in image_entries:
        try:
            results = run_ocr(await _download(att))
        except Exception as e:
            print(f"[scan] Error on {att.filename}: {e}")
            continue

        mv = extract_market_value(results)
        if mv is not None:
            market_value = mv

        deposit_amounts.extend(extract_deposit_amounts(results))

    loot_str = f"{market_value:,}" if market_value is not None else "not found"
    silver_str = f"{sum(deposit_amounts):,}" if deposit_amounts else "not found"

    view = ScanConfigView(loot_str, silver_str)
    await ctx.send(view._build_status(), view=view)

    try:
        await asyncio.wait_for(view._done.wait(), timeout=180)
    except asyncio.TimeoutError:
        await ctx.send("Timed out. Run `!scan` again.")
        return

    loot_val = market_value or 0
    silver_val = sum(deposit_amounts) if deposit_amounts else 0
    loot_key = "damaged-loot-total" if view.is_damaged else "non-damaged-loot-total"
    loot_label = "Damaged loot" if view.is_damaged else "Non-damaged loot"

    summary = (
        f"**Split Summary**\n"
        f"Caller: {view.caller.mention}\n"
        f"Split Type: {view.split_type}\n"
        f"Event Type: {view.event_type}\n"
        f"{loot_label}: {loot_val:,} silver\n"
        f"Silver Bags: {silver_val:,} silver\n\n"
        f"▶ `/split-loot loot-split-type:{view.split_type} caller-name:{view.caller.display_name} "
        f"event-type:{view.event_type} {loot_key}:{loot_val} silver-bags-total:{silver_val}`"
    )
    await ctx.send(summary)


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    bot.run(DISCORD_TOKEN)
