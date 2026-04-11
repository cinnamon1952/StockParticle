from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from notebook_runner import execute_notebook_cell
from notebook_sources import LPPL_NOTEBOOK_SOURCE, MULTIFRACTAL_NOTEBOOK_SOURCES
from plot_helpers import PROJECT_ROOT, pretty_label, relative_to_project
from section_featured import build_featured_section
from section_lppl import build_lppl_section
from section_multifractal import build_multifractal_section


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DashboardState:
    def __init__(self) -> None:
        self._asset_map: dict[str, Path] = {}
        self._next_id = 1
        self._lock = threading.Lock()
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

    def reload(self) -> None:
        with self._lock:
            self._asset_map = {}
            self._next_id = 1
            self.manifest = self._build_manifest()
            self.manifest_json = json.dumps(self.manifest).encode("utf-8")

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "title": "StockParticle Plot Dashboard",
            "multifractal": build_multifractal_section(self.register_asset),
            "featured": build_featured_section(self.register_asset),
            "lppl": build_lppl_section(self.register_asset),
        }


def parse_lppl_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tickers_raw = payload.get("tickers", "")
    if isinstance(tickers_raw, list):
        tickers = [str(value).strip() for value in tickers_raw if str(value).strip()]
    else:
        tickers = [value.strip() for value in str(tickers_raw).split(",") if value.strip()]
    if not tickers:
        raise ValueError("Provide at least one ticker.")

    offsets_raw = payload.get("window_start_offsets", "")
    if isinstance(offsets_raw, list):
        offsets = [int(value) for value in offsets_raw]
    else:
        offsets = [int(value.strip()) for value in str(offsets_raw).split(",") if value.strip()]
    if not offsets:
        raise ValueError("Provide at least one LPPL window offset.")

    return {
        "tickers": tickers,
        "start_data": str(payload.get("start_data", "2007-01-01")),
        "end_data": str(payload.get("end_data", "2009-06-01")),
        "target_date": str(payload.get("target_date", "2008-09-28")),
        "window_start_offsets": offsets,
        "roll_days": int(payload.get("roll_days", 60)),
        "roll_step": int(payload.get("roll_step", 5)),
        "plot_duration": int(payload.get("plot_duration", 150)),
        "surrogate_n": int(payload.get("surrogate_n", 100)),
    }


def parse_multifractal_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    group = str(payload.get("group", "")).strip()
    if group not in MULTIFRACTAL_NOTEBOOK_SOURCES:
        raise ValueError("Unknown multifractal notebook group.")

    overrides: dict[str, Any] = {
        "OUT_DIR": MULTIFRACTAL_NOTEBOOK_SOURCES[group]["output_dir"].name,
    }
    if payload.get("tickers"):
        overrides["TICKERS"] = [value.strip() for value in str(payload["tickers"]).split(",") if value.strip()]
    if payload.get("start"):
        overrides["START"] = str(payload["start"])
    if payload.get("end"):
        overrides["END"] = str(payload["end"])
    if payload.get("sectors"):
        overrides["SECTORS"] = payload["sectors"]

    return group, overrides


def html_shell() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StockParticle</title>
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
      --dim:#617084;
      --line:rgba(255,255,255,0.06);
      --blue:#4f9cf7;
      --purple:#8b5cf6;
      --teal:#2dd4bf;
      --pink:#ec4899;
      --green:#16a34a;
      --danger:#ef4444;
      --r:10px;
    }
    html,body{height:100%}
    body{
      background:var(--bg);
      color:var(--ink);
      font-family:Inter,system-ui,-apple-system,sans-serif;
      font-size:14px;
      line-height:1.55;
      display:flex;
      height:100vh;
      overflow:hidden;
    }
    button,select,input{font:inherit;color:inherit}
    .sidebar{
      width:240px;
      min-width:240px;
      background:var(--nav);
      border-right:1px solid var(--line);
      display:flex;
      flex-direction:column;
    }
    .sidebar-brand{
      padding:24px 20px 20px;
      border-bottom:1px solid var(--line);
    }
    .sidebar-brand h1{font-size:1.05rem;font-weight:700;color:var(--bright)}
    .sidebar-brand span{font-size:0.75rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.04em}
    .sidebar-nav{padding:12px 10px;flex:1}
    .nav-label{font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--dim);padding:14px 12px 6px}
    .nav-item{
      display:flex;align-items:center;gap:10px;width:100%;border:0;background:transparent;color:var(--dim);
      padding:9px 12px;border-radius:8px;cursor:pointer;font-size:0.88rem;font-weight:500;text-align:left;
    }
    .nav-item:hover{background:rgba(255,255,255,0.04);color:var(--ink)}
    .nav-item.active{background:rgba(79,156,247,0.1);color:var(--blue)}
    .nav-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .dot-blue{background:var(--blue)}
    .dot-purple{background:var(--purple)}
    .dot-teal{background:var(--teal)}
    .main{flex:1;overflow-y:auto;background:var(--bg)}

    #landing{
      position:fixed;inset:0;z-index:200;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      text-align:center;padding:48px 32px;
      background:var(--bg);
      background-image:
        radial-gradient(ellipse 70% 50% at 50% 0%, rgba(79,156,247,0.08), transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 100%, rgba(139,92,246,0.06), transparent 60%),
        radial-gradient(ellipse 35% 45% at 10% 80%, rgba(45,212,191,0.05), transparent 60%);
      transition:opacity .5s ease, transform .5s ease;
    }
    #landing.out{opacity:0;transform:scale(1.03);pointer-events:none}
    .hero-icon{
      width:80px;height:80px;border-radius:22px;
      background:linear-gradient(135deg, rgba(79,156,247,0.15), rgba(139,92,246,0.15));
      border:1px solid rgba(255,255,255,0.08);
      display:flex;align-items:center;justify-content:center;
      margin-bottom:32px;animation:hero-float 6s ease-in-out infinite;
    }
    @keyframes hero-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
    .hero-icon svg{width:40px;height:40px;fill:none;stroke:var(--blue);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
    .hero-title{
      font-size:clamp(2.4rem,5vw,3.8rem);font-weight:800;letter-spacing:-0.04em;
      color:var(--bright);line-height:1.1;margin-bottom:18px;
    }
    .hero-title span{
      background:linear-gradient(135deg,var(--blue),var(--purple));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;
    }
    .hero-sub{
      font-size:1.1rem;color:var(--dim);max-width:54ch;line-height:1.75;margin-bottom:48px;
    }
    .hero-cards{
      display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px;
      max-width:960px;width:100%;
    }
    .hero-card{
      background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px 24px;
      text-align:left;cursor:pointer;
      transition:border-color .25s, transform .25s, box-shadow .25s;
    }
    .hero-card:hover{
      border-color:rgba(79,156,247,0.22);transform:translateY(-5px);
      box-shadow:0 20px 48px -12px rgba(0,0,0,0.5);
    }
    .hero-card-dot{width:10px;height:10px;border-radius:50%;margin-bottom:14px}
    .hero-card h3{font-size:1.04rem;font-weight:600;color:var(--bright);margin-bottom:8px}
    .hero-card p{font-size:0.88rem;color:var(--dim);line-height:1.6}
    .hero-card .hero-card-count{
      display:inline-block;margin-top:14px;font-size:0.78rem;font-weight:600;
      color:var(--ink);background:rgba(255,255,255,0.05);padding:4px 10px;border-radius:6px;
    }
    .hero-footer{
      margin-top:52px;display:flex;gap:36px;flex-wrap:wrap;justify-content:center;
    }
    .hero-stat{text-align:center}
    .hero-stat-value{font-size:1.9rem;font-weight:700;color:var(--bright);letter-spacing:-0.03em}
    .hero-stat-label{font-size:0.74rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;margin-top:3px}
    .hero-tagline{
      margin-top:56px;font-size:0.82rem;color:var(--dim);letter-spacing:0.02em;
    }
    .hero-tagline kbd{
      display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:5px;
      background:var(--surface);font-size:0.78rem;font-family:inherit;margin:0 2px;
    }
    .topbar{
      position:sticky;top:0;z-index:20;background:rgba(6,8,13,0.9);backdrop-filter:blur(16px);
      border-bottom:1px solid var(--line);padding:16px 32px;display:flex;align-items:center;justify-content:space-between;gap:16px;
    }
    .topbar h2{font-size:1rem;font-weight:600;color:var(--bright)}
    .topbar-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .topbar-controls label,.field label{font-size:0.75rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.05em;font-weight:600}
    select,input{
      border:1px solid var(--line);background:var(--surface);padding:8px 12px;border-radius:8px;outline:none;
    }
    input[type="date"]{min-width:160px}
    .content{padding:28px 32px 64px}
    .section-desc{color:var(--dim);font-size:0.92rem;max-width:72ch;margin-bottom:24px;line-height:1.65}
    .subtle{color:var(--dim);font-size:0.84rem}
    .grid{display:grid;gap:20px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
    .card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;transition:border-color .2s, transform .2s ease}
    .card:hover{border-color:rgba(79,156,247,0.18);transform:translateY(-3px)}
    .card-button{display:block;width:100%;border:0;padding:0;background:transparent;text-align:left;cursor:pointer;color:inherit}
    .card-img-wrap{position:relative;overflow:hidden}
    .card-img-wrap img{display:block;width:100%;height:auto}
    .card-body{padding:14px 16px 16px}
    .card-body h3{font-size:0.92rem;font-weight:600;color:var(--bright);margin-bottom:6px}
    .card-explanation{color:var(--dim);font-size:0.84rem;line-height:1.55}
    .empty,.loading{color:var(--dim);padding:28px 0}
    .group-block{margin-bottom:28px}
    .group-block h3{font-size:0.98rem;color:var(--bright);margin-bottom:6px}
    .toolbar-form{
      margin:20px 0 28px;padding:18px;border:1px solid var(--line);border-radius:12px;background:var(--surface);
      display:grid;gap:14px;
    }
    .field-grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
    .field{display:grid;gap:6px}
    .field-wide{grid-column:1 / -1}
    .form-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
    .button-primary{
      border:1px solid rgba(79,156,247,0.35);background:rgba(79,156,247,0.16);color:#dbeafe;padding:9px 14px;border-radius:8px;cursor:pointer;font-weight:600;
    }
    .button-primary[disabled]{opacity:.65;cursor:progress}
    .status-ok{color:#86efac}
    .status-error{color:#fca5a5}
    .details-section{margin-top:28px;border-top:1px solid var(--line);padding-top:20px}
    .details-toggle{
      display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);background:var(--surface);color:var(--dim);
      padding:8px 16px;border-radius:8px;cursor:pointer;font-size:0.85rem;font-weight:500;margin-bottom:18px;
    }
    .details-grid{display:none}
    .details-grid.open{display:grid}
    .modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;padding:32px;background:rgba(3,4,8,0.9);z-index:100}
    .modal.open{display:flex}
    .modal-dialog{
      width:min(1140px,100%);max-height:calc(100vh - 64px);overflow-y:auto;background:var(--surface);border:1px solid rgba(255,255,255,0.07);border-radius:14px;
    }
    .modal-header{display:flex;justify-content:space-between;align-items:start;padding:20px 24px 14px;border-bottom:1px solid var(--line)}
    .modal-header h2{font-size:1.05rem;font-weight:600;color:var(--bright)}
    .modal-close{border:1px solid var(--line);background:var(--card);padding:6px 14px;border-radius:7px;cursor:pointer;font-size:0.82rem;color:var(--dim)}
    .modal-body{padding:20px 24px 24px}
    .modal-body img{display:block;width:100%;height:auto;border-radius:8px;border:1px solid var(--line);margin-bottom:18px}
    .modal-explanation{color:var(--ink);font-size:0.94rem;line-height:1.65;max-width:72ch}
    @media(max-width:900px){
      body{flex-direction:column}
      .sidebar{width:100%;min-width:100%;border-right:0;border-bottom:1px solid var(--line)}
      .content{padding:20px 16px 48px}
      .topbar{padding:14px 16px;align-items:start;flex-direction:column}
    }
  </style>
</head>
<body>
  <div id="landing">
    <div class="hero-icon">
      <svg viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M7 16l4-8 4 5 5-9"/></svg>
    </div>
    <h1 class="hero-title">Stock<span>Particle</span></h1>
    <p class="hero-sub">
      Multifractal analysis, LPPL crash diagnostics, and correlation structure
      for equities, commodities, and crypto &mdash; driven directly from Jupyter notebooks.
    </p>
    <div class="hero-cards" id="hero-cards"></div>
    <div class="hero-footer" id="hero-footer"></div>
    <p class="hero-tagline">Select a section above to explore, or use the sidebar anytime.</p>
  </div>

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
    <div class="topbar">
      <h2 id="topbar-title">StockParticle</h2>
      <div class="topbar-controls" id="topbar-controls"></div>
    </div>
    <div class="content" id="app"><div class="loading" style="text-align:center;padding:120px 20px;color:var(--dim)">Loading dashboard&hellip;</div></div>
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
    const nav = document.getElementById("sidebar-nav");
    const topbarTitle = document.getElementById("topbar-title");
    const topbarControls = document.getElementById("topbar-controls");
    const app = document.getElementById("app");
    const modal = document.getElementById("modal");
    const modalTitle = document.getElementById("modal-title");
    const modalImage = document.getElementById("modal-image");
    const modalExplanation = document.getElementById("modal-explanation");
    const modalClose = document.getElementById("modal-close");

    const TABS = [
      {id:"multifractal", label:"Multifractal", dot:"dot-blue"},
      {id:"featured", label:"Attached Plots", dot:"dot-purple"},
      {id:"lppl", label:"LPPL-MFDFA", dot:"dot-teal"},
    ];

    const esc = (v) => String(v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

    const sanitizeTicker = (value) => String(value).replace(/^\\^/, "").replace(/=/g, "_");

    const landing = document.getElementById("landing");
    const heroCardsEl = document.getElementById("hero-cards");
    const heroFooterEl = document.getElementById("hero-footer");
    let landingVisible = true;

    let manifest = null;
    let activeTab = "multifractal";
    let activeMfGroup = "mf_plots";
    let activeLpplTicker = "";
    let mfStatus = "";
    let mfStatusClass = "";
    let mfRunning = false;
    let lpplStatus = "";
    let lpplStatusClass = "";
    let lpplRunning = false;

    function card(image) {
      const payload = esc(JSON.stringify(image));
      return `<article class="card"><button class="card-button" type="button" data-image="${payload}"><div class="card-img-wrap"><img src="${image.src}" alt="${esc(image.title)}"></div><div class="card-body"><h3>${esc(image.title)}</h3><div class="card-explanation">${esc(image.explanation || "")}</div></div></button></article>`;
    }

    const empty = (msg) => `<div class="empty">${esc(msg)}</div>`;
    const grid = (images) => images.length ? `<div class="grid">${images.map(card).join("")}</div>` : empty("No plots found.");
    const sourceInfo = (source) => source ? `<div class="subtle">Notebook source: <code>${esc(source.notebook)}</code> cell <code>${esc(source.cell_index)}</code> -> <code>${esc(source.output_dir)}</code>${source.legacy_cell_index !== undefined ? `, legacy LPPL cell <code>${esc(source.legacy_cell_index)}</code>` : ""}</div>` : "";

    function buildNav() {
      const items = TABS.map((tab) =>
        `<button class="nav-item ${tab.id===activeTab?"active":""}" data-tab="${tab.id}" type="button"><span class="nav-dot ${tab.dot}"></span>${esc(tab.label)}</button>`
      ).join("");
      nav.innerHTML = `<div class="nav-label">Views</div>${items}`;
      nav.querySelectorAll(".nav-item").forEach((button) => {
        button.addEventListener("click", () => {
          activeTab = button.dataset.tab;
          render();
        });
      });
    }

    function mountCards() {
      document.querySelectorAll(".card-button").forEach((button) => {
        button.addEventListener("click", () => {
          const image = JSON.parse(button.dataset.image);
          modalTitle.textContent = image.title;
          modalImage.src = image.src;
          modalImage.alt = image.title;
          modalExplanation.textContent = image.explanation || "";
          modal.classList.add("open");
          modal.setAttribute("aria-hidden", "false");
        });
      });
    }

    function closeModal() {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
      modalImage.src = "";
    }

    function countImages(obj) {
      if (!obj) return 0;
      if (Array.isArray(obj.images)) return obj.images.length;
      let total = 0;
      if (obj.groups) { for (const k of Object.keys(obj.groups)) total += countImages(obj.groups[k]); }
      if (obj.tickers) { for (const k of Object.keys(obj.tickers)) { const e = obj.tickers[k]; total += (e.highlights||[]).length + (e.residuals||[]).length; } }
      return total;
    }

    function dismissLanding(tab) {
      activeTab = tab;
      landingVisible = false;
      landing.classList.add("out");
      landing.addEventListener("transitionend", () => { landing.style.display = "none"; }, {once:true});
      render();
    }

    function populateLanding() {
      const mfCount = countImages(manifest.multifractal);
      const featCount = countImages(manifest.featured);
      const lpplCount = countImages(manifest.lppl);
      const totalPlots = mfCount + featCount + lpplCount;
      const lpplTickers = Object.keys(manifest.lppl.tickers || {}).length;

      heroCardsEl.innerHTML = `
        <div class="hero-card" data-goto="multifractal">
          <div class="hero-card-dot dot-blue"></div>
          <h3>Multifractal Analysis</h3>
          <p>MF-DFA return structure and symmetric Pearson correlation across sectors and asset classes.</p>
          <span class="hero-card-count">${mfCount} plots</span>
        </div>
        <div class="hero-card" data-goto="lppl">
          <div class="hero-card-dot dot-teal"></div>
          <h3>LPPL-MFDFA</h3>
          <p>Log-periodic power law fits with rolling multifractal residual diagnostics and surrogate tests.</p>
          <span class="hero-card-count">${lpplTickers} tickers &middot; ${lpplCount} plots</span>
        </div>
        <div class="hero-card" data-goto="featured">
          <div class="hero-card-dot dot-purple"></div>
          <h3>Featured Plots</h3>
          <p>Standalone reference figures including drawdown distributions and pairwise correlation heatmaps.</p>
          <span class="hero-card-count">${featCount} plots</span>
        </div>`;

      heroFooterEl.innerHTML = `
        <div class="hero-stat">
          <div class="hero-stat-value">${totalPlots}</div>
          <div class="hero-stat-label">Total Plots</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-value">3</div>
          <div class="hero-stat-label">Notebook Cells</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-value">${lpplTickers}</div>
          <div class="hero-stat-label">LPPL Tickers</div>
        </div>`;

      heroCardsEl.querySelectorAll(".hero-card[data-goto]").forEach((card) => {
        card.addEventListener("click", () => dismissLanding(card.dataset.goto));
      });
    }

    function renderFeatured() {
      topbarTitle.textContent = manifest.featured.title;
      topbarControls.innerHTML = "";
      app.innerHTML = `<p class="section-desc">${esc(manifest.featured.description)}</p>${grid(manifest.featured.images)}`;
    }

    function renderMultifractal() {
      const groups = manifest.multifractal.groups;
      const keys = Object.keys(groups);
      if (!keys.includes(activeMfGroup)) {
        activeMfGroup = keys[0] || "mf_plots";
      }
      const group = groups[activeMfGroup];
      topbarTitle.textContent = manifest.multifractal.title;
      topbarControls.innerHTML = `<label for="mf-group-select">Source</label><select id="mf-group-select">${keys.map((key) => `<option value="${esc(key)}" ${key===activeMfGroup?"selected":""}>${esc(groups[key].title)}</option>`).join("")}</select>`;
      const mfStatusHtml = mfStatus ? `<div class="${esc(mfStatusClass)}">${esc(mfStatus)}</div>` : `<div class="subtle">Use refresh to rerun the exact notebook cell that owns this section.</div>`;
      app.innerHTML = `<p class="section-desc">${esc(manifest.multifractal.description)}</p><div class="toolbar-form"><div class="form-actions"><button class="button-primary" id="mf-refresh" type="button" ${mfRunning ? "disabled" : ""}>${mfRunning ? "Running..." : "Refresh From Notebook Cell"}</button>${mfStatusHtml}</div>${sourceInfo(group.source)}</div><div class="group-block"><h3>${esc(group.title)}</h3><p class="section-desc">${esc(group.description)}</p>${grid(group.images)}</div>`;
      const select = document.getElementById("mf-group-select");
      if (select) {
        select.addEventListener("change", () => {
          activeMfGroup = select.value;
          renderMultifractal();
          mountCards();
        });
      }
      const refreshButton = document.getElementById("mf-refresh");
      if (refreshButton) {
        refreshButton.addEventListener("click", async () => {
          mfRunning = true;
          mfStatus = `Running ${group.source.notebook} cell ${group.source.cell_index}...`;
          mfStatusClass = "subtle";
          renderMultifractal();
          mountCards();
          try {
            const response = await fetch("/api/multifractal/run", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({group: activeMfGroup}),
            });
            const result = await response.json();
            if (!response.ok) {
              throw new Error(result.error || "Multifractal refresh failed.");
            }
            manifest = result.manifest;
            mfStatus = `Notebook cell ran. ${result.changed_count} files changed in ${result.output_dir}.`;
            mfStatusClass = "status-ok";
          } catch (error) {
            mfStatus = error.message || "Multifractal refresh failed.";
            mfStatusClass = "status-error";
          } finally {
            mfRunning = false;
            renderMultifractal();
            mountCards();
          }
        });
      }
    }

    function lpplForm(defaults) {
      const statusHtml = lpplStatus ? `<div class="${esc(lpplStatusClass)}">${esc(lpplStatus)}</div>` : `<div class="subtle">Run the current LPPL notebook cell here, then the tab refreshes from the output folder automatically.</div>`;
      return `
        <form id="lppl-form" class="toolbar-form">
          <div class="field-grid">
            <div class="field field-wide">
              <label for="tickers">Tickers</label>
              <input id="tickers" name="tickers" type="text" value="${esc(defaults.tickers.join(","))}">
            </div>
            <div class="field">
              <label for="start_data">Start Date</label>
              <input id="start_data" name="start_data" type="date" value="${esc(defaults.start_data)}">
            </div>
            <div class="field">
              <label for="end_data">End Date</label>
              <input id="end_data" name="end_data" type="date" value="${esc(defaults.end_data)}">
            </div>
            <div class="field">
              <label for="target_date">Target Date</label>
              <input id="target_date" name="target_date" type="date" value="${esc(defaults.target_date)}">
            </div>
            <div class="field">
              <label for="window_start_offsets">Window Offsets</label>
              <input id="window_start_offsets" name="window_start_offsets" type="text" value="${esc(defaults.window_start_offsets.join(","))}">
            </div>
            <div class="field">
              <label for="roll_days">Roll Days</label>
              <input id="roll_days" name="roll_days" type="number" value="${esc(defaults.roll_days)}">
            </div>
            <div class="field">
              <label for="roll_step">Roll Step</label>
              <input id="roll_step" name="roll_step" type="number" value="${esc(defaults.roll_step)}">
            </div>
            <div class="field">
              <label for="plot_duration">Plot Duration</label>
              <input id="plot_duration" name="plot_duration" type="number" value="${esc(defaults.plot_duration)}">
            </div>
            <div class="field">
              <label for="surrogate_n">Surrogates</label>
              <input id="surrogate_n" name="surrogate_n" type="number" value="${esc(defaults.surrogate_n)}">
            </div>
          </div>
          <div class="form-actions">
            <button class="button-primary" id="lppl-submit" type="submit" ${lpplRunning ? "disabled" : ""}>${lpplRunning ? "Running..." : "Run LPPL Generation"}</button>
            ${statusHtml}
          </div>
          ${sourceInfo(manifest.lppl.source)}
        </form>
      `;
    }

    function renderLppl() {
      const tickers = Object.keys(manifest.lppl.tickers);
      if (!activeLpplTicker || !manifest.lppl.tickers[activeLpplTicker]) {
        activeLpplTicker = tickers[0] || "";
      }

      topbarTitle.textContent = manifest.lppl.title;
      if (tickers.length) {
        topbarControls.innerHTML = `<label for="lppl-select">Ticker</label><select id="lppl-select">${tickers.map((ticker) => `<option value="${esc(ticker)}" ${ticker===activeLpplTicker?"selected":""}>${esc(ticker)}</option>`).join("")}</select>`;
      } else {
        topbarControls.innerHTML = "";
      }

      const defaults = manifest.lppl.defaults;
      const entry = activeLpplTicker ? manifest.lppl.tickers[activeLpplTicker] : null;
      const highlights = entry ? grid(entry.highlights) : empty("No LPPL plots found.");
      const residuals = entry && entry.residuals.length
        ? `<div class="details-section"><button class="details-toggle" type="button" id="res-toggle">Residual diagnostics (${entry.residuals.length})</button><div class="details-grid grid" id="res-grid">${entry.residuals.map(card).join("")}</div></div>`
        : "";

      app.innerHTML = `<p class="section-desc">${esc(manifest.lppl.description)}</p>${lpplForm(defaults)}${entry ? highlights : empty("No LPPL plots found yet.")}${residuals}`;

      const select = document.getElementById("lppl-select");
      if (select) {
        select.addEventListener("change", () => {
          activeLpplTicker = select.value;
          renderLppl();
          mountCards();
        });
      }

      const toggle = document.getElementById("res-toggle");
      const residualGrid = document.getElementById("res-grid");
      if (toggle && residualGrid) {
        toggle.addEventListener("click", () => residualGrid.classList.toggle("open"));
      }

      bindLpplForm();
    }

    function bindLpplForm() {
      const form = document.getElementById("lppl-form");
      if (!form) return;
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const data = new FormData(form);
        const payload = Object.fromEntries(data.entries());
        lpplRunning = true;
        lpplStatus = `Running ${manifest.lppl.source.notebook} cell ${manifest.lppl.source.cell_index}...`;
        lpplStatusClass = "subtle";
        renderLppl();
        mountCards();
        try {
          const response = await fetch("/api/lppl/run", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
          });
          const result = await response.json();
          if (!response.ok) {
            throw new Error(result.error || "LPPL generation failed.");
          }
          manifest = result.manifest;
          const requested = String(payload.tickers || "").split(",").map((value) => value.trim()).filter(Boolean);
          const preferredTicker = requested.length ? sanitizeTicker(requested[0]) : activeLpplTicker;
          if (manifest.lppl.tickers[preferredTicker]) {
            activeLpplTicker = preferredTicker;
          }
          lpplStatus = `Notebook cell ran. ${result.changed_count} files changed in ${result.output_dir}.`;
          lpplStatusClass = "status-ok";
        } catch (error) {
          lpplStatus = error.message || "LPPL generation failed.";
          lpplStatusClass = "status-error";
        } finally {
          lpplRunning = false;
          renderLppl();
          mountCards();
        }
      });
    }

    function render() {
      if (!manifest) return;
      buildNav();
      switch (activeTab) {
        case "multifractal":
          renderMultifractal();
          break;
        case "featured":
          renderFeatured();
          break;
        case "lppl":
          renderLppl();
          break;
      }
      mountCards();
    }

    modalClose.addEventListener("click", closeModal);
    modal.addEventListener("click", (event) => { if (event.target === modal) closeModal(); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape" && modal.classList.contains("open")) closeModal(); });

    fetch("/api/manifest")
      .then((response) => {
        if (!response.ok) throw new Error("Load failed");
        return response.json();
      })
      .then((data) => {
        manifest = data;
        populateLanding();
        render();
      })
      .catch((error) => {
        app.innerHTML = `<div class="empty">Could not load data. ${esc(error.message)}</div>`;
        landing.classList.add("out");
      });
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "StockParticleDashboard/2.0"

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

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/lppl/run", "/api/multifractal/run"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")

            if parsed.path == "/api/multifractal/run":
                group, overrides = parse_multifractal_payload(payload)
                source = MULTIFRACTAL_NOTEBOOK_SOURCES[group]
                result = execute_notebook_cell(
                    notebook_path=source["notebook"],
                    cell_index=int(source["cell_index"]),
                    output_dir=source["output_dir"],
                    assignment_overrides=overrides,
                )
            else:
                config_kwargs = parse_lppl_payload(payload)
                lppl_overrides = {
                    "tickers": config_kwargs["tickers"],
                    "START_DATA": config_kwargs["start_data"],
                    "END_DATA": config_kwargs["end_data"],
                    "TARGET_DATE": config_kwargs["target_date"],
                    "window_start_offsets": config_kwargs["window_start_offsets"],
                    "ROLL_DAYS": config_kwargs["roll_days"],
                    "ROLL_STEP": config_kwargs["roll_step"],
                    "plot_duration": config_kwargs["plot_duration"],
                    "SURROGATE_N": config_kwargs["surrogate_n"],
                    "OUT_DIR": LPPL_NOTEBOOK_SOURCE["output_dir"].name,
                }
                result = execute_notebook_cell(
                    notebook_path=LPPL_NOTEBOOK_SOURCE["notebook"],
                    cell_index=int(LPPL_NOTEBOOK_SOURCE["cell_index"]),
                    output_dir=LPPL_NOTEBOOK_SOURCE["output_dir"],
                    assignment_overrides=lppl_overrides,
                )

            self.server.state.reload()
            response = {
                "ok": True,
                "changed_count": result["changed_count"],
                "changed_files": result["changed_files"],
                "output_dir": result["output_dir"],
                "cell_index": result["cell_index"],
                "manifest": self.server.state.manifest,
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(HTTPStatus.OK)
        except Exception as exc:
            body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
    parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard automatically.")
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
        print("\\nStopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
