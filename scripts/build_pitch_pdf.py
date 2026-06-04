"""Render the pitch deck to Pitch_YMusic.pdf (6 slides, 16:9) via headless Chromium.

LibreOffice/PowerPoint were unavailable on the build machine, so the deck content
is reproduced as styled HTML slides and printed to PDF. Content mirrors
Pitch_YMusic.pptx (criteria answers + repo link).

Run:  uv run --with playwright python scripts/build_pitch_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "Pitch_YMusic.pdf"
REPO = "github.com/ak1232320/nndl_capstone_2026"

CSS = """
@page { size: 13.333in 7.5in; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1b2733; }
.slide { width: 13.333in; height: 7.5in; padding: 0.55in 0.75in 0.7in; position: relative;
         page-break-after: always; overflow: hidden; }
.slide:last-child { page-break-after: auto; }
.kicker { color: #d95f0e; font-weight: 700; letter-spacing: 1.5px; font-size: 14pt; }
.title { font-size: 27pt; font-weight: 800; margin: 4pt 0 14pt; line-height: 1.1; }
.lead { font-size: 14.5pt; color: #34414f; margin: 0 0 10pt; }
.label { color: #c4561a; font-weight: 700; font-size: 13.5pt; margin: 12pt 0 2pt; }
ul { margin: 4pt 0 0; padding-left: 20pt; }
li { font-size: 13pt; line-height: 1.5; margin: 4pt 0; }
.note { color: #d95f0e; font-weight: 600; font-size: 12.5pt; margin-top: 12pt; }
.footer { position: absolute; left: 0.75in; right: 0.75in; bottom: 0.34in; display: flex;
          justify-content: space-between; font-size: 9.5pt; color: #8a97a3;
          border-top: 1px solid #e4e9ef; padding-top: 6pt; }
mark { background: #fff1e8; color: #c4561a; font-weight: 700; padding: 0 3px; }
/* title slide */
.cover { display: flex; flex-direction: column; justify-content: center; height: 100%; }
.cover .big { font-size: 40pt; font-weight: 800; margin: 2pt 0 6pt; }
.cover .sub { font-size: 16pt; color: #34414f; }
.cover .team { margin-top: 22pt; font-size: 14pt; }
.cover .meta { font-size: 12.5pt; color: #66727e; margin-top: 2pt; }
.badge { display: inline-block; background: #fff1e8; color: #c4561a; font-weight: 700;
         padding: 5pt 12pt; border-radius: 7pt; margin-top: 20pt; font-size: 13pt; }
.repo { color: #2c7fb8; font-weight: 700; margin-top: 12pt; font-size: 13.5pt; }
"""


def footer(n: int) -> str:
    return f'<div class="footer"><span>Next-Gen Music Recommender · Capstone Pitch</span><span>0{n} / 05</span></div>'


SLIDES = [
    # 1 — cover
    f"""<div class="slide"><div class="cover">
      <div class="kicker">CAPSTONE PROJECT PITCH</div>
      <div class="big">Next-Gen Music Recommender</div>
      <div class="sub">on real Yandex listening data (Yambda)</div>
      <div class="team">Anna Grishkina · Valeria Karpova · Aleksey Kosychev</div>
      <div class="meta">HSE Master's · NNDL · Module 2</div>
      <span class="badge">Real-world operational data (+2 bonus)</span>
      <div class="repo">{REPO}</div>
    </div></div>""",
    # 2 — problem
    f"""<div class="slide">
      <div class="kicker">01 / PROBLEM STATEMENT</div>
      <div class="title">Predict the next track for a streaming user</div>
      <div class="label">What we are solving</div>
      <div class="lead">Given a 10 M-track catalogue, predict the next track a user will actually
        want to hear — the central decision a streaming product makes every few minutes.</div>
      <div class="label">Why now</div>
      <ul>
        <li>≈ 70 % of streaming users struggle to discover new music — bad recommendations are the dominant churn driver.</li>
        <li>The streaming market is saturated; subscriber growth has plateaued, so retention is the #1 commercial metric.</li>
        <li>A few % lift in next-track ranking separates catching up from leading.</li>
      </ul>{footer(1)}</div>""",
    # 3 — data
    f"""<div class="slide">
      <div class="kicker">02 / DATA — OUR KEY ADVANTAGE</div>
      <div class="title">Yambda · 4.79 B real listening events from Yandex Music</div>
      <div class="label">Source</div>
      <ul>
        <li>Yambda by Yandex on Hugging Face — released May 2025 (arXiv:2505.22238, RecSys '25).</li>
        <li>1 M users · 9.39 M tracks · 4.79 B events <b>(we train on the 50M slice)</b>.</li>
      </ul>
      <div class="label">Why it is strong for this task</div>
      <ul>
        <li>Three signal layers: explicit (likes/dislikes), implicit (plays/skips/replays), and 128-d audio embeddings (82 % of tracks).</li>
        <li>An <code>is_organic</code> flag on every event separates user-driven from recommender-driven plays — trains without the recommender-bias loop.</li>
        <li>Real production distributions: heavy-tail catalogue, real cold-start, real session structure.</li>
      </ul>
      <div class="note">Real-world operational data → qualifies for the +2 bonus, target = 10 / 10.</div>{footer(2)}</div>""",
    # 4 — modeling
    f"""<div class="slide">
      <div class="kicker">03 / MODELING APPROACH</div>
      <div class="title">Hybrid: SASRec sequence model + audio-content tower</div>
      <div class="label">A · SASRec — sequence</div>
      <ul>
        <li>Causal-attention Transformer over the user's listening history (structurally a small GPT).</li>
        <li>Learns transition patterns and predicts the next track from the sequence so far.</li>
      </ul>
      <div class="label">B · Content tower</div>
      <ul>
        <li>Scores tracks by audio-taste similarity on Yandex 128-d embeddings — complementary to the sequence model.</li>
      </ul>
      <div class="label">Fusion & why this approach</div>
      <ul>
        <li>Outputs fused with a <b>validation-tuned weight (β)</b> — late, not joint, fusion (joint overfit).</li>
        <li>Pure CF loses sequence; pure content ignores behaviour — the hybrid handles both.</li>
      </ul>{footer(3)}</div>""",
    # 5 — KPI
    f"""<div class="slide">
      <div class="kicker">04 / KPI & RESULTS</div>
      <div class="title">NDCG@10 — ranking quality at the top of the list</div>
      <div class="label">Primary metric</div>
      <div class="lead">NDCG@10 under the dataset's Global Temporal Split, ranking over the full
        629 k-item catalogue (the same protocol as the Yandex paper).</div>
      <div class="label">Results (our harness)</div>
      <ul>
        <li>Best baselines: ItemKNN 0.071 · <b>SASRec 0.0735</b> (98 % of the paper, beats all baselines).</li>
        <li><mark>Hybrid: +9.7 % ± 2.0 % over SASRec</mark> — robust across 5 held-out user splits.</li>
      </ul>
      <div class="label">Expected business impact</div>
      <ul>
        <li>+9.7 % better next-track ranking → lever on listening time and churn.</li>
        <li>Delivered on free data + one free GPU (near-zero infrastructure cost).</li>
      </ul>{footer(4)}</div>""",
    # 6 — summary
    f"""<div class="slide">
      <div class="kicker">05 / SUMMARY & CALL TO ACTION</div>
      <div class="title">What we ship</div>
      <div class="lead">Real Yandex listening data (Yambda-50M) → hybrid neural recommender:
        SASRec + audio-content tower, fused with a validation-tuned weight → predict the next track.
        <b>SASRec beats every baseline and reproduces the paper to 98 %; late fusion adds a robust
        +9.7 % ± 2.0 % NDCG@10</b> across 5 held-out splits.</div>
      <div class="label">What it took / how to reproduce</div>
      <ul>
        <li>Yambda dataset on Hugging Face — public, free.</li>
        <li>One free Kaggle T4 (16 GB) GPU — no paid infrastructure.</li>
        <li>Reproducible: <span class="repo">{REPO}</span> (pipeline notebooks 00–07).</li>
      </ul>{footer(5)}</div>""",
]


def main() -> None:
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(SLIDES)}</body></html>"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(path=str(DST), width="13.333in", height="7.5in", print_background=True)
        browser.close()
    print(f"wrote {DST} ({DST.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
