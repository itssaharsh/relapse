"""Draws space/og.png, the demo page's preview image, from space/data/stage2.json (the numbers the page itself shows).

    python scripts/make_og.py
Needs Pillow and the DejaVu fonts. The headline is the page's own h1.
"""
from __future__ import annotations

import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = "/usr/share/fonts/truetype/dejavu/"
INK, MUTED, PAPER, CARD, RED, GREEN, RED_BG, GREEN_BG = "#141414", "#4a4a45", "#faf8f3", "#ffffff", "#b3261e", "#0b6b4f", "#f7d9d5", "#d5efe5"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONTS + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def wrap(draw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if draw.textlength((line + " " + word).strip(), font=f) <= width:
            line = (line + " " + word).strip()
        else:
            lines.append(line); line = word
    return lines + [line]


def main() -> None:
    b = json.load(open(os.path.join(ROOT, "space", "data", "stage2.json")))["brief"]
    page = open(os.path.join(ROOT, "space", "index.html")).read()
    headline = re.search(r"<h1>(.*?)</h1>", page, re.S).group(1).strip()
    im = Image.new("RGB", (1200, 630), PAPER); d = ImageDraw.Draw(im)
    d.rectangle((72, 22, 84, 32), fill=RED); d.rectangle((86, 22, 100, 32), fill=INK); d.text((112, 16), "RELAPSE", font=font(19, True), fill=INK)
    d.line((0, 54, 1200, 54), fill="#e4e1d8", width=2)
    y = 84
    for line in wrap(d, headline, font(38, True), 1060):
        d.text((72, y), line, font=font(38, True), fill=INK); y += 46
    y += 12
    for line in wrap(d, "Unlearned and checked through an outdated tokenizer setting, then run through the official one: runs whose hazardous knowledge came back", font(21, True), 1060):
        d.text((72, y), line, font=font(21, True), fill=MUTED); y += 28
    top = max(y + 16, 318)
    cards = ((72, RED, RED_BG, "SafeTune's default layers", "(layers 5 to 7)", b["default"], "Came back",
              f"{b['default']['mean_rise']:+.1f} points; {b['default']['mmlu_audited']:.1f} MMLU points lost as checked"),
             (612, GREEN, GREEN_BG, "Layers 3 to 5", "(on four of the five models)", b["released"], "Held",
              f"{b['released']['mean_rise']:+.1f} points; {b['released']['mmlu_audited']:.1f} MMLU points lost as checked"))
    for x, colour, tint, title, note, col, tag, line in cards:
        d.rounded_rectangle((x, top, x + 516, 602), radius=10, fill=CARD, outline="#e4e1d8")
        d.rectangle((x, top, x + 516, top + 6), fill=colour)
        d.text((x + 24, top + 22), title, font=font(20, True), fill=INK)
        d.text((x + 24 + d.textlength(title + " ", font=font(20, True)), top + 23), note, font=font(19), fill=MUTED)
        big = str(col["tok_relapses"]); d.text((x + 24, top + 50), big, font=font(104, True), fill=colour)
        d.text((x + 36 + d.textlength(big, font=font(104, True)), top + 108), f"of {col['tok_runs']}", font=font(44, True), fill=colour)
        w = d.textlength(tag, font=font(15, True)) + 20
        d.rounded_rectangle((x + 24, top + 178, x + 24 + w, top + 202), radius=12, fill=tint); d.text((x + 34, top + 181), tag, font=font(15, True), fill=colour)
        d.text((x + 34 + w, top + 181), line, font=font(15), fill=INK)
        for i in range(col["tok_runs"]):
            box = (x + 24 + i * 28, top + 222, x + 44 + i * 28, top + 242)
            d.rectangle(box, fill=colour if i < col["tok_relapses"] else CARD, outline=colour, width=2)
    im.save(os.path.join(ROOT, "space", "og.png"))


if __name__ == "__main__":
    main()
