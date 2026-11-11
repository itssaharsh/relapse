"""Render docs/ARTICLE.md as the one-page static Space that hosts the write-up on Hugging Face.

    python scripts/build_article_space.py <out dir>      # then upload the folder as a static Space
The page template lives next to the output as index.html; this script only swaps in the rendered article body.
"""
from __future__ import annotations

import os
import re
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(out: str) -> None:
    src = re.sub(r"<!-- \w+:(start|end) -->\n?", "", open(os.path.join(ROOT, "docs", "ARTICLE.md")).read())
    body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    page = open(os.path.join(out, "index.html")).read()
    page = re.sub(r"<main>.*</main>", lambda m: "<main>" + body + "</main>", page, flags=re.S)
    open(os.path.join(out, "index.html"), "w").write(page)


if __name__ == "__main__":
    main(sys.argv[1])
