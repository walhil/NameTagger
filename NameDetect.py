import asyncio

import aiohttp
import discord
from discord.ext import commands

from config import DISCORD_TOKEN
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


@bot.command()
async def scan(ctx: commands.Context):
    """Scan loot and deposit images posted after the last !ping and output /split-loot arguments."""
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

    lines = []

    if market_value is not None:
        lines.append(f"**Est. Market Value:** {market_value:,} silver")
    else:
        lines.append("**Est. Market Value:** not found")

    if deposit_amounts:
        total_dep = sum(deposit_amounts)
        if len(deposit_amounts) == 1:
            lines.append(f"**Total Deposited:** {total_dep:,} silver")
        else:
            breakdown = " + ".join(f"{a:,}" for a in deposit_amounts)
            lines.append(f"**Total Deposited:** {total_dep:,} silver ({breakdown})")
    else:
        lines.append("**Total Deposited:** not found")

    if market_value is not None and deposit_amounts:
        lines.append(f"\n▶ `/split-loot {market_value} {sum(deposit_amounts)}`")

    await ctx.send("\n".join(lines))


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable is not set.")
    bot.run(DISCORD_TOKEN)
