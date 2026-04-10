from __future__ import annotations

import argparse
import json
import mimetypes
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = PROJECT_ROOT / "scripts" / "trial"
ATTACHED_IMAGE_DEFAULTS = [
    {
        "title": "2008 Crash Rolling Pairwise Correlation",
        "caption": "DJIA price and pairwise-correlation view around the 2008 bubble and crash.",
        "prefix": "fig5_bubble_correlation",
    },
    {
        "title": "NASDAQ Drawdown Distribution",
        "caption": "Drawdown distribution against the null-hypothesis line for NASDAQ (USA).",
        "prefix": "drawdown_nasdaq_usa_",
    },
    {
        "title": "DJIA Correlation Heatmaps",
        "caption": "Component correlation heatmaps across pre-bubble, pre-crash, and crash regimes.",
        "prefix": "fig5b_pairwise_heatmaps",
    },
]


def natural_sort_key(value: str) -> list[Any]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", value)]


def pretty_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: natural_sort_key(path.name),
    )


def cursor_asset_dir() -> Path:
    project_slug = str(PROJECT_ROOT.resolve()).strip("/").replace("/", "-")
    return Path.home() / ".cursor" / "projects" / project_slug / "assets"


def resolve_featured_image(prefix: str) -> Path | None:
    search_dirs = [
        PROJECT_ROOT / "dashboard" / "assets",
        PROJECT_ROOT / "assets",
        cursor_asset_dir(),
    ]

    for directory in search_dirs:
        if not directory.exists():
            continue
        matches = sorted(
            [path for path in directory.glob(f"{prefix}*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda path: natural_sort_key(path.name),
        )
        if matches:
            return matches[0]

    return None


def explain_mf_corr_plot(path: Path) -> str:
    name = path.stem
    if "fullrange" in name:
        return "This is the widest sector-level view, so it helps show whether elevated co-movement is just a crisis blip or something more persistent."
    if "sector" in name:
        return "This compares correlation behavior across sectors. The useful thing to watch is whether normally distinct sectors begin moving more like one synchronized block."
    if "symmetric" in name or "fq" in name:
        return "This plot looks at how correlation behaves across q-values, which roughly means across calmer versus more extreme fluctuations."
    return "This multifractal correlation plot highlights how strongly market components move together across scales."


def explain_featured_plot(title: str) -> str:
    if "2008 Crash" in title:
        return "The story here is rising lockstep behavior before and during the crash. When correlations climb, diversification tends to get less helpful exactly when risk is highest."
    if "Drawdown" in title:
        return "This compares observed drawdowns with a benchmark null line. The tail deviations are the interesting part because they suggest stress that is more severe than ordinary randomness would imply."
    if "Heatmaps" in title:
        return "These heatmaps are a nice visual shortcut for systemic risk: darker and more uniform coloring usually means more names are moving together."
    return "This featured chart was included as a standalone reference plot for the dashboard."


def explain_lppl_plot(ticker: str, suffix: str) -> str:
    if suffix == "overview":
        return f"This is the clearest top-level summary for `{ticker}` because it puts the main LPPL-MFDFA regime story in one frame."
    if suffix == "rolling_trajectory":
        return f"This rolling trajectory is useful for seeing how the signal evolves through time, rather than only where it ends up."
    if suffix.endswith("residuals_surrogate"):
        window = suffix.split("_", 1)[0].upper()
        return f"This `{window}` residual-surrogate plot for `{ticker}` is more of a diagnostic check than a headline plot. It helps judge whether the leftover structure after fitting still looks too organized."
    return f"This LPPL-MFDFA plot is part of the diagnostic set for `{ticker}`."


def explain_fractal_mfdfa(stock: str) -> str:
    return f"This MFDFA plot for `{stock}` summarizes how scaling behavior changes across fluctuation levels. More spread or curvature usually points to richer multifractal structure."


class DashboardState:
    def __init__(self) -> None:
        self._asset_map: dict[str, Path] = {}
        self._next_id = 1
        self.manifest = self._build_manifest()
        self.manifest_json = json.dumps(self.manifest).encode("utf-8")

    def register_asset(
        self,
        path: Path,
        *,
        title: str | None = None,
        caption: str | None = None,
        explanation: str | None = None,
    ) -> dict[str, str]:
        asset_id = f"asset-{self._next_id}"
        self._next_id += 1
        resolved = path.resolve()
        self._asset_map[asset_id] = resolved
        return {
            "id": asset_id,
            "src": f"/asset/{asset_id}",
            "title": title or pretty_label(path.stem),
            "caption": caption or relative_to_project(path),
            "explanation": explanation or "",
            "path": relative_to_project(path),
            "file_name": path.name,
        }

    def get_asset(self, asset_id: str) -> Path | None:
        return self._asset_map.get(asset_id)

    def _build_manifest(self) -> dict[str, Any]:
        mf_corr_dir = TRIAL_DIR / "mf_corr_plots"
        lppl_dir = TRIAL_DIR / "lppl_mfdfa_v2_plots"
        fractal_dir = TRIAL_DIR / "fractal_structure_results"

        return {
            "title": "StockParticle Plot Dashboard",
            "mf_corr": {
                "title": "MF Correlation Plots",
                "description": "All plots found under scripts/trial/mf_corr_plots.",
                "images": [
                    self.register_asset(
                        path,
                        title=pretty_label(path.stem),
                        caption=relative_to_project(path),
                        explanation=explain_mf_corr_plot(path),
                    )
                    for path in image_files(mf_corr_dir)
                ],
            },
            "featured": {
                "title": "Featured Attached Plots",
                "description": "The three standalone plots provided for this dashboard.",
                "images": self._build_featured_images(),
            },
            "lppl": {
                "title": "LPPL-MFDFA Highlights",
                "description": (
                    "Curated to emphasize the two most informative plot types per ticker: the overall "
                    "LPPL overview and the rolling trajectory. Residual-surrogate diagnostics are available "
                    "as optional supporting plots."
                ),
                "tickers": self._build_lppl_groups(lppl_dir),
            },
            "fractal": {
                "title": "Fractal Structure Results",
                "description": (
                    "Choose a stock to inspect all of its available fractal-structure output plots, grouped "
                    "by analysis type."
                ),
                "stocks": self._build_fractal_groups(fractal_dir),
            },
        }

    def _build_featured_images(self) -> list[dict[str, str]]:
        featured = []
        for item in ATTACHED_IMAGE_DEFAULTS:
            image_path = resolve_featured_image(str(item["prefix"]))
            if image_path is not None:
                featured.append(
                    self.register_asset(
                        image_path,
                        title=str(item["title"]),
                        caption=str(item["caption"]),
                        explanation=explain_featured_plot(str(item["title"])),
                    )
                )
        return featured

    def _build_lppl_groups(self, lppl_dir: Path) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}

        for path in image_files(lppl_dir):
            stem = path.stem
            if "_" not in stem:
                continue
            ticker, suffix = stem.split("_", 1)
            entry = grouped.setdefault(ticker, {"highlights": [], "residuals": []})

            if suffix in {"overview", "rolling_trajectory"}:
                entry["highlights"].append(
                    self.register_asset(
                        path,
                        title=f"{ticker} {pretty_label(suffix)}",
                        caption=relative_to_project(path),
                        explanation=explain_lppl_plot(ticker, suffix),
                    )
                )
            elif suffix.endswith("residuals_surrogate"):
                window = suffix.split("_", 1)[0]
                entry["residuals"].append(
                    self.register_asset(
                        path,
                        title=f"{ticker} Residual Diagnostics {window.upper()}",
                        caption=relative_to_project(path),
                        explanation=explain_lppl_plot(ticker, suffix),
                    )
                )

        for ticker, data in grouped.items():
            data["highlights"].sort(key=lambda item: 0 if "Overview" in item["title"] else 1)
            data["residuals"].sort(key=lambda item: natural_sort_key(item["file_name"]))
            data["summary"] = f"{ticker} LPPL and MFDFA view"

        return dict(sorted(grouped.items(), key=lambda item: natural_sort_key(item[0])))

    def _build_fractal_groups(self, fractal_dir: Path) -> dict[str, Any]:
        grouped: dict[str, Any] = {}

        if not fractal_dir.exists():
            return grouped

        for stock_dir in sorted([path for path in fractal_dir.iterdir() if path.is_dir()], key=lambda path: natural_sort_key(path.name)):
            mfdfa_dir = stock_dir / "mfdfa"
            images = [
                self.register_asset(
                    path,
                    title=pretty_label(path.stem),
                    caption=relative_to_project(path),
                    explanation=explain_fractal_mfdfa(stock_dir.name),
                )
                for path in image_files(mfdfa_dir)
            ]
            grouped[stock_dir.name] = {
                "label": stock_dir.name,
                "images": images,
                "summary": f"MFDFA output for {stock_dir.name}",
            }

        return grouped


def html_shell() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StockParticle</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}

    :root{
      --bg:#06080d;
      --nav:#0a0d14;
      --surface:#0e1219;
      --card:#12161f;
      --card-hover:#171c28;
      --ink:#b6beca;
      --bright:#e2e8f0;
      --dim:#4a5568;
      --line:rgba(255,255,255,0.05);
      --blue:#4f9cf7;
      --purple:#8b5cf6;
      --teal:#2dd4bf;
      --pink:#ec4899;
      --orange:#f59e0b;
      --r:10px;
    }

    html{height:100%}

    body{
      background:var(--bg);
      color:var(--ink);
      font-family:'Inter',system-ui,-apple-system,sans-serif;
      font-size:14px;
      line-height:1.55;
      -webkit-font-smoothing:antialiased;
      display:flex;
      height:100vh;
      overflow:hidden;
    }

    button,select,input{font:inherit;color:inherit}

    ::selection{background:rgba(79,156,247,0.25)}

    /* ── sidebar ── */
    .sidebar{
      width:240px;
      min-width:240px;
      background:var(--nav);
      border-right:1px solid var(--line);
      display:flex;
      flex-direction:column;
      padding:0;
      overflow-y:auto;
    }

    .sidebar-brand{
      padding:24px 20px 20px;
      border-bottom:1px solid var(--line);
    }

    .sidebar-brand h1{
      font-size:1.05rem;
      font-weight:700;
      letter-spacing:-0.03em;
      color:var(--bright);
      margin-bottom:2px;
    }

    .sidebar-brand span{
      font-size:0.75rem;
      color:var(--dim);
      letter-spacing:0.04em;
      text-transform:uppercase;
    }

    .sidebar-nav{
      padding:12px 10px;
      flex:1;
    }

    .nav-label{
      font-size:0.68rem;
      font-weight:600;
      text-transform:uppercase;
      letter-spacing:0.08em;
      color:var(--dim);
      padding:14px 12px 6px;
    }

    .nav-item{
      display:flex;
      align-items:center;
      gap:10px;
      width:100%;
      border:0;
      background:transparent;
      color:var(--dim);
      padding:9px 12px;
      border-radius:8px;
      cursor:pointer;
      font-size:0.88rem;
      font-weight:500;
      transition:all .15s ease;
      text-align:left;
    }

    .nav-item:hover{
      background:rgba(255,255,255,0.04);
      color:var(--ink);
    }

    .nav-item.active{
      background:rgba(79,156,247,0.1);
      color:var(--blue);
    }

    .nav-dot{
      width:8px;height:8px;
      border-radius:50%;
      flex-shrink:0;
    }

    .dot-blue{background:var(--blue)}
    .dot-purple{background:var(--purple)}
    .dot-teal{background:var(--teal)}
    .dot-pink{background:var(--pink)}

    /* ── main content ── */
    .main{
      flex:1;
      overflow-y:auto;
      background:
        radial-gradient(ellipse 70% 30% at 30% 0%, rgba(79,156,247,0.04), transparent 70%),
        radial-gradient(ellipse 50% 30% at 85% 100%, rgba(139,92,246,0.03), transparent 70%),
        var(--bg);
    }

    .topbar{
      position:sticky;
      top:0;
      z-index:20;
      background:rgba(6,8,13,0.85);
      backdrop-filter:blur(16px);
      -webkit-backdrop-filter:blur(16px);
      border-bottom:1px solid var(--line);
      padding:16px 32px;
      display:flex;
      align-items:center;
      justify-content:space-between;
    }

    .topbar h2{
      font-size:1rem;
      font-weight:600;
      color:var(--bright);
    }

    .topbar-controls{
      display:flex;
      gap:10px;
      align-items:center;
    }

    .topbar-controls label{
      font-size:0.78rem;
      color:var(--dim);
      text-transform:uppercase;
      letter-spacing:0.05em;
      font-weight:600;
    }

    select{
      border:1px solid var(--line);
      background:var(--surface);
      padding:7px 12px;
      border-radius:8px;
      outline:none;
      transition:border-color .2s;
      font-size:0.88rem;
    }

    select:focus{border-color:var(--blue)}

    .content{
      padding:28px 32px 64px;
    }

    .section-desc{
      color:var(--dim);
      font-size:0.92rem;
      max-width:64ch;
      margin-bottom:24px;
      line-height:1.65;
    }

    /* ── card grid ── */
    .grid{
      display:grid;
      gap:20px;
      grid-template-columns:repeat(auto-fill,minmax(360px,1fr));
    }

    .card{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:var(--r);
      overflow:hidden;
      position:relative;
      transition:border-color .2s, transform .2s ease, box-shadow .2s ease;
    }

    .card:hover{
      border-color:rgba(79,156,247,0.18);
      transform:translateY(-4px) scale(1.005);
      box-shadow:0 20px 40px -12px rgba(0,0,0,0.5);
    }

    .card-button{
      display:block;
      width:100%;
      border:0;padding:0;
      background:transparent;
      text-align:left;
      cursor:pointer;
      color:inherit;
    }

    .card-img-wrap{
      position:relative;
      overflow:hidden;
    }

    .card-img-wrap img{
      display:block;
      width:100%;
      height:auto;
      transition:transform .3s ease;
    }

    .card:hover .card-img-wrap img{
      transform:scale(1.03);
    }

    .card-overlay{
      position:absolute;
      inset:0;
      background:linear-gradient(180deg, transparent 40%, rgba(6,8,13,0.85) 100%);
      opacity:0;
      transition:opacity .25s ease;
      display:flex;
      align-items:flex-end;
      padding:16px;
    }

    .card:hover .card-overlay{
      opacity:1;
    }

    .card-overlay span{
      display:inline-flex;
      align-items:center;
      gap:6px;
      background:rgba(255,255,255,0.12);
      backdrop-filter:blur(12px);
      -webkit-backdrop-filter:blur(12px);
      padding:7px 14px;
      border-radius:20px;
      font-size:0.82rem;
      font-weight:500;
      color:#fff;
    }

    .card-overlay svg{
      width:14px;height:14px;
      fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;
    }

    .card-body{
      padding:14px 16px 16px;
    }

    .card-body h3{
      font-size:0.92rem;
      font-weight:600;
      color:var(--bright);
      margin-bottom:6px;
    }

    .card-explanation{
      color:var(--dim);
      font-size:0.84rem;
      line-height:1.55;
    }

    /* ── details toggle ── */
    .details-section{
      margin-top:32px;
      border-top:1px solid var(--line);
      padding-top:22px;
    }

    .details-toggle{
      display:inline-flex;
      align-items:center;
      gap:8px;
      border:1px solid var(--line);
      background:var(--surface);
      color:var(--dim);
      padding:8px 16px;
      border-radius:8px;
      cursor:pointer;
      font-size:0.85rem;
      font-weight:500;
      margin-bottom:18px;
      transition:all .15s;
    }

    .details-toggle:hover{
      border-color:rgba(255,255,255,0.12);
      color:var(--ink);
    }

    .details-toggle svg{
      width:14px;height:14px;
      fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;
      transition:transform .2s;
    }

    .details-toggle.open svg{
      transform:rotate(180deg);
    }

    .details-grid{
      display:none;
    }

    .details-grid.open{
      display:grid;
    }

    .empty{color:var(--dim);padding:20px 0}

    .loading{color:var(--dim);padding:48px 0}

    /* ── modal ── */
    .modal{
      position:fixed;inset:0;
      display:none;
      align-items:center;justify-content:center;
      padding:32px;
      background:rgba(3,4,8,0.9);
      backdrop-filter:blur(16px);
      -webkit-backdrop-filter:blur(16px);
      z-index:100;
    }

    .modal.open{display:flex}

    .modal-dialog{
      width:min(1140px,100%);
      max-height:calc(100vh - 64px);
      overflow-y:auto;
      background:var(--surface);
      border:1px solid rgba(255,255,255,0.07);
      border-radius:14px;
      box-shadow:0 32px 80px rgba(0,0,0,0.6);
      animation:modalIn .2s ease;
    }

    @keyframes modalIn{
      from{opacity:0;transform:scale(0.97) translateY(8px)}
      to{opacity:1;transform:scale(1) translateY(0)}
    }

    .modal-header{
      display:flex;
      justify-content:space-between;
      align-items:start;
      padding:20px 24px 14px;
      border-bottom:1px solid var(--line);
    }

    .modal-header h2{
      font-size:1.05rem;
      font-weight:600;
      color:var(--bright);
    }

    .modal-close{
      border:1px solid var(--line);
      background:var(--card);
      padding:6px 14px;
      border-radius:7px;
      cursor:pointer;
      font-size:0.82rem;
      font-weight:500;
      transition:all .12s;
      flex-shrink:0;
      color:var(--dim);
    }

    .modal-close:hover{
      background:var(--card-hover);
      color:var(--bright);
      border-color:rgba(255,255,255,0.12);
    }

    .modal-body{padding:20px 24px 24px}

    .modal-body img{
      display:block;
      width:100%;height:auto;
      border-radius:8px;
      border:1px solid var(--line);
      margin-bottom:18px;
    }

    .modal-explanation{
      color:var(--ink);
      font-size:0.94rem;
      line-height:1.65;
      max-width:72ch;
    }

    ::-webkit-scrollbar{width:7px;height:7px}
    ::-webkit-scrollbar-track{background:transparent}
    ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:4px}
    ::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.14)}

    @media(max-width:800px){
      body{flex-direction:column}
      .sidebar{width:100%;min-width:100%;height:auto;flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid var(--line)}
      .sidebar-brand{display:none}
      .sidebar-nav{display:flex;gap:4px;padding:8px;flex:0}
      .nav-label{display:none}
      .content{padding:20px 16px 48px}
      .grid{grid-template-columns:1fr}
      .modal{padding:12px}
    }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-brand">
      <h1>StockParticle</h1>
      <span>Plot Dashboard</span>
    </div>
    <nav class="sidebar-nav" id="sidebar-nav">
      <div class="nav-label">Views</div>
    </nav>
  </aside>

  <div class="main">
    <div class="topbar" id="topbar">
      <h2 id="topbar-title">Loading...</h2>
      <div class="topbar-controls" id="topbar-controls"></div>
    </div>
    <div class="content" id="app"><div class="loading">Loading plots...</div></div>
  </div>

  <div class="modal" id="modal" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-header">
        <h2 id="modal-title"></h2>
        <button class="modal-close" id="modal-close" type="button">Close</button>
      </div>
      <div class="modal-body">
        <img id="modal-image" alt="">
        <div class="modal-explanation" id="modal-explanation"></div>
      </div>
    </div>
  </div>

  <script>
    const EXPAND_ICON = `<svg viewBox="0 0 24 24"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

    const nav = document.getElementById("sidebar-nav");
    const topbarTitle = document.getElementById("topbar-title");
    const topbarControls = document.getElementById("topbar-controls");
    const app = document.getElementById("app");
    const modal = document.getElementById("modal");
    const modalTitle = document.getElementById("modal-title");
    const modalImage = document.getElementById("modal-image");
    const modalExplanation = document.getElementById("modal-explanation");
    const modalClose = document.getElementById("modal-close");

    const esc = (v) => String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");

    function card(image) {
      const p = esc(JSON.stringify(image));
      return `<article class="card"><button class="card-button" type="button" data-image="${p}"><div class="card-img-wrap"><img src="${image.src}" alt="${esc(image.title)}"><div class="card-overlay"><span>${EXPAND_ICON} Expand</span></div></div><div class="card-body"><h3>${esc(image.title)}</h3><div class="card-explanation">${esc(image.explanation||"")}</div></div></button></article>`;
    }

    const empty = (msg) => `<div class="empty">${esc(msg)}</div>`;
    const grid = (images) => images.length ? `<div class="grid">${images.map(card).join("")}</div>` : empty("No plots found.");

    const TABS = [
      {id:"mf-corr", label:"MF Correlation", dot:"dot-blue"},
      {id:"featured", label:"Attached Plots", dot:"dot-purple"},
      {id:"lppl", label:"LPPL-MFDFA", dot:"dot-teal"},
      {id:"fractal", label:"Fractal MFDFA", dot:"dot-pink"},
    ];

    let manifest = null;
    let activeTab = "mf-corr";

    function buildNav() {
      const items = TABS.map(t =>
        `<button class="nav-item ${t.id===activeTab?"active":""}" data-tab="${t.id}" type="button"><span class="nav-dot ${t.dot}"></span>${esc(t.label)}</button>`
      ).join("");
      nav.innerHTML = `<div class="nav-label">Views</div>${items}`;
      nav.querySelectorAll(".nav-item").forEach(b => {
        b.addEventListener("click", () => { activeTab = b.dataset.tab; render(); });
      });
    }

    function render() {
      if (!manifest) return;
      buildNav();

      switch(activeTab) {
        case "mf-corr":
          topbarTitle.textContent = "MF Correlation Plots";
          topbarControls.innerHTML = "";
          app.innerHTML = `<p class="section-desc">${esc(manifest.mf_corr.description)}</p>${grid(manifest.mf_corr.images)}`;
          break;
        case "featured":
          topbarTitle.textContent = "Attached Plots";
          topbarControls.innerHTML = "";
          app.innerHTML = `<p class="section-desc">${esc(manifest.featured.description)}</p>${grid(manifest.featured.images)}`;
          break;
        case "lppl":
          renderLppl();
          break;
        case "fractal":
          renderFractal();
          break;
      }

      mountCards();
    }

    function renderLppl() {
      const tickers = Object.keys(manifest.lppl.tickers);
      topbarTitle.textContent = "LPPL-MFDFA Highlights";

      if (!tickers.length) {
        topbarControls.innerHTML = "";
        app.innerHTML = empty("No LPPL-MFDFA plots found.");
        return;
      }

      const sel = topbarControls.querySelector("#lppl-select");
      const current = sel ? sel.value : tickers[0];
      topbarControls.innerHTML = `<label for="lppl-select">Ticker</label><select id="lppl-select">${tickers.map(t=>`<option value="${esc(t)}" ${t===current?"selected":""}>${esc(t)}</option>`).join("")}</select>`;

      const entry = manifest.lppl.tickers[current];
      const hl = grid(entry.highlights);
      const res = entry.residuals.length
        ? `<div class="details-section"><button class="details-toggle" type="button" id="res-toggle"><span>Residual diagnostics (${entry.residuals.length})</span><svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg></button><div class="details-grid grid" id="res-grid">${entry.residuals.map(card).join("")}</div></div>`
        : "";

      app.innerHTML = `<p class="section-desc">${esc(manifest.lppl.description)}</p>${hl}${res}`;

      document.getElementById("lppl-select").addEventListener("change", () => { renderLppl(); mountCards(); });

      const toggle = document.getElementById("res-toggle");
      const resGrid = document.getElementById("res-grid");
      if (toggle && resGrid) {
        toggle.addEventListener("click", () => {
          toggle.classList.toggle("open");
          resGrid.classList.toggle("open");
        });
      }
    }

    function renderFractal() {
      const stocks = Object.keys(manifest.fractal.stocks);
      topbarTitle.textContent = "Fractal MFDFA";

      if (!stocks.length) {
        topbarControls.innerHTML = "";
        app.innerHTML = empty("No fractal structure results found.");
        return;
      }

      const sel = topbarControls.querySelector("#fractal-select");
      const current = sel ? sel.value : stocks[0];
      topbarControls.innerHTML = `<label for="fractal-select">Stock</label><select id="fractal-select">${stocks.map(s=>`<option value="${esc(s)}" ${s===current?"selected":""}>${esc(s)}</option>`).join("")}</select>`;

      const entry = manifest.fractal.stocks[current];
      app.innerHTML = `<p class="section-desc">${esc(manifest.fractal.description)}</p>${grid(entry.images)}`;

      document.getElementById("fractal-select").addEventListener("change", () => { renderFractal(); mountCards(); });
    }

    function mountCards() {
      document.querySelectorAll(".card-button").forEach(b => {
        b.addEventListener("click", () => {
          const img = JSON.parse(b.dataset.image);
          modalTitle.textContent = img.title;
          modalImage.src = img.src;
          modalImage.alt = img.title;
          modalExplanation.textContent = img.explanation || "";
          modal.classList.add("open");
          modal.setAttribute("aria-hidden","false");
        });
      });
    }

    function closeModal() {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden","true");
      modalImage.src = "";
    }

    modalClose.addEventListener("click", closeModal);
    modal.addEventListener("click", e => { if(e.target===modal) closeModal(); });
    document.addEventListener("keydown", e => { if(e.key==="Escape" && modal.classList.contains("open")) closeModal(); });

    fetch("/api/manifest")
      .then(r => { if(!r.ok) throw new Error("Load failed"); return r.json(); })
      .then(data => { manifest = data; render(); })
      .catch(err => { app.innerHTML = `<div class="empty">Could not load data. ${esc(err.message)}</div>`; });
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "StockParticleDashboard/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            body = html_shell().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/manifest":
            body = self.server.state.manifest_json
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path.startswith("/asset/"):
            asset_id = parsed.path.rsplit("/", 1)[-1]
            asset_path = self.server.state.get_asset(asset_id)
            if asset_path is None or not asset_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return

            try:
                data = asset_path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not read asset")
                return

            content_type, _ = mimetypes.guess_type(asset_path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: DashboardState) -> None:
        super().__init__(server_address, DashboardHandler)
        self.state = state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the StockParticle plot dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on. Default: 8000")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the dashboard automatically in the default browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = DashboardState()
    server = DashboardServer((args.host, args.port), state)
    url = f"http://{args.host}:{args.port}"

    print(f"Serving StockParticle dashboard at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
