"""Render REPORT.md to REPORT.pdf (Markdown -> HTML -> headless-Chromium print).

Figures are inlined as base64 so relative paths never break. No system tools
needed beyond the Chromium that Playwright downloads.

Run:
    uv run --with markdown --with playwright bash -c \
        "python -m playwright install chromium && python scripts/build_pdf.py"
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REPORT.md"
DST = ROOT / "REPORT.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 21pt; margin: 0 0 4pt; border-bottom: 2px solid #d95f0e; padding-bottom: 4pt; }
h2 { font-size: 14pt; margin: 16pt 0 6pt; color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; color: #34495e; }
p, li { margin: 4pt 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #ccc; padding: 3pt 7pt; text-align: left; }
th { background: #f2f4f6; }
tr:nth-child(even) td { background: #fafbfc; }
code { background: #f2f4f6; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; padding: 8pt; border-radius: 5px; overflow-x: auto; font-size: 8.5pt; }
pre code { background: none; padding: 0; }
img { max-width: 92%; display: block; margin: 8pt auto; }
blockquote { border-left: 3px solid #d95f0e; margin: 8pt 0; padding: 2pt 12pt; color: #444; background: #fff8f2; }
a { color: #2c7fb8; text-decoration: none; }
h2, h3 { page-break-after: avoid; }
table, img, pre { page-break-inside: avoid; }
"""


def inline_images(html: str, base: Path) -> str:
    def repl(m: re.Match) -> str:
        src = m.group(1)
        data = base64.b64encode((base / src).read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'
    return re.sub(r'src="([^"]+\.png)"', repl, html)


def main() -> None:
    body = markdown.markdown(
        SRC.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = inline_images(body, ROOT)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(path=str(DST), format="A4", print_background=True)
        browser.close()
    print(f"wrote {DST} ({DST.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
