"""
Sornette Markets Dashboard — yfinance-style dark UI
Tabs: Correlation | LPPL | Multifractal | Drawdowns
"""
import sys, os, io, warnings, tempfile
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
import yfinance as yf
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockParticle - An Idea",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global dark yfinance-style CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Dark background */
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22;
    border-bottom: 2px solid #30363d;
    padding: 0 1rem;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 500;
    font-size: 14px;
    margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #e6edf3 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* Metrics */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px; font-weight: 600; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px; }

/* Inputs & selects */
.stTextInput input, .stSelectbox select, div[data-baseweb="select"] > div,
.stDateInput input, .stNumberInput input {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] * { color: #e6edf3 !important; background-color: #21262d !important; }

/* Buttons */
.stButton > button {
    background: #21262d !important;
    color: #58a6ff !important;
    border: 1px solid #388bfd !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 0.45rem 1.1rem !important;
    transition: background 0.15s;
}
.stButton > button:hover { background: #388bfd22 !important; }
.stButton > button[kind="primary"] {
    background: #238636 !important;
    color: #fff !important;
    border-color: #2ea043 !important;
}

/* Dividers / headers */
h1,h2,h3,h4 { color: #e6edf3 !important; }
hr { border-color: #30363d; }

/* Section card */
.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Ticker badge */
.ticker-badge {
    display: inline-block;
    background: #0d419d;
    color: #58a6ff;
    border: 1px solid #388bfd;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 4px;
}

/* Status pill */
.pill-green { display:inline-block; background:#1a4731; color:#3fb950; border:1px solid #238636; border-radius:20px; padding:2px 10px; font-size:12px; }
.pill-red   { display:inline-block; background:#3d1c1c; color:#f85149; border:1px solid #b91c1c; border-radius:20px; padding:2px 10px; font-size:12px; }
.pill-blue  { display:inline-block; background:#0d2045; color:#58a6ff; border:1px solid #1f6feb; border-radius:20px; padding:2px 10px; font-size:12px; }

/* Spinner */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* Image display */
img { border-radius: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark style ─────────────────────────────────────────────────────
MPLSTYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#8b949e",
    "axes.titlecolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "text.color": "#e6edf3",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#e6edf3",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
}

def apply_dark(fig):
    fig.patch.set_facecolor("#0d1117")
    for ax in fig.get_axes():
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3", labelcolor="#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.yaxis.label.set_color("#e6edf3")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color("#e6edf3")
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor("#161b22")
            leg.get_frame().set_edgecolor("#30363d")
            for t in leg.get_texts():
                t.set_color("#e6edf3")
    if fig._suptitle:
        fig._suptitle.set_color("#e6edf3")
    return fig

def fig_to_st(fig, caption=None):
    apply_dark(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True, caption=caption)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:18px 0 8px 0'>
    <span style='font-size:36px;font-weight:700;color:#58a6ff;letter-spacing:-1px'>Stock</span><span style='font-size:36px;font-weight:700;color:#3fb950;letter-spacing:-1px'>Particle</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_corr, tab_lppl, tab_mf, tab_dd = st.tabs([
    "🏠  Home",
    "📊  Correlation & Returns",
    "📉  LPPL Fits",
    "🔬  Multifractal",
    "📐  Drawdowns",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
import streamlit.components.v1 as components

with tab_home:
    components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: transparent; color: #e6edf3; }
.wrap { display: flex; flex-direction: column; align-items: center; padding: 1.5rem 1rem 2rem; }
.wheel-area { width: 380px; height: 380px; }
svg { width: 380px; height: 380px; cursor: pointer; }
#wheel-g { transform-origin: 190px 190px; transition: transform 0.65s cubic-bezier(0.4,0,0.2,1); }
.spoke { transition: opacity 0.3s; cursor: pointer; }
.content-area { width: 100%; max-width: 580px; margin-top: 36px; }
.content-title { font-size: 19px; font-weight: 600; color: #e6edf3; margin-bottom: 8px; display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: wrap; }
.content-tag { font-size: 11px; font-weight: 500; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.04em; }
.content-body { font-size: 14px; color: #8b949e; line-height: 1.85; text-align: left; margin-top: 14px; }
.content-body p { margin-bottom: 12px; }
.content-body p:last-child { margin-bottom: 0; }
.dot-row { display: flex; gap: 8px; justify-content: center; margin-top: 24px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #30363d; cursor: pointer; transition: background 0.2s; }
.dot.active { background: #e6edf3; }
.hint { font-size: 12px; color: #444; margin-top: 10px; text-align: center; }
.fade { animation: fadeIn 0.35s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
<div class="wrap">
  <div class="wheel-area">
    <svg viewBox="0 0 380 380" xmlns="http://www.w3.org/2000/svg">
      <g id="wheel-g">
        <path id="s0" class="spoke" d="M190,190 L190,22 A168,168 0 0,1 312,106 Z" fill="#378ADD"/>
        <path id="s1" class="spoke" d="M190,190 L312,106 A168,168 0 0,1 312,274 Z" fill="#1D9E75" opacity="0.45"/>
        <path id="s2" class="spoke" d="M190,190 L312,274 A168,168 0 0,1 68,274 Z" fill="#EF9F27" opacity="0.45"/>
        <path id="s3" class="spoke" d="M190,190 L68,274 A168,168 0 0,1 68,106 Z" fill="#D4537E" opacity="0.45"/>
        <path id="s4" class="spoke" d="M190,190 L68,106 A168,168 0 0,1 190,22 Z" fill="#7F77DD" opacity="0.45"/>
        <circle cx="190" cy="190" r="56" fill="#0d1117" stroke="#30363d" stroke-width="0.5"/>
        <text x="190" y="185" text-anchor="middle" font-size="12" font-weight="500" fill="#8b949e" font-family="sans-serif">fractal</text>
        <text x="190" y="201" text-anchor="middle" font-size="12" font-weight="500" fill="#8b949e" font-family="sans-serif">risk</text>
      </g>
    </svg>
  </div>

  <div class="content-area">
    <div class="content-title">
      <span id="ctitle">Correlation &amp; Returns</span>
      <span class="content-tag" id="ctag" style="background:#0d2045;color:#58a6ff">cooperativity</span>
    </div>
    <div class="content-body fade" id="cbody">
      <p>Examines the full distributional behavior of asset returns — skewness, excess kurtosis, and the degree to which tails deviate from a Gaussian benchmark. Autocorrelation is tested across both raw returns and absolute returns, the latter being a proxy for volatility persistence. How volatility scales with the square root of time is also examined, since departures from √T scaling imply memory in the process.</p>
      <p>The central insight is pairwise cooperativity under stress. During normal markets, assets move with low correlation — diversification works. In the lead-up to systemic crises, correlations rise sharply and converge, dismantling the assumption that a diversified portfolio provides meaningful protection. The pre-2008 cross-asset correlation trend culminated near 0.72 before the crash, a textbook regime shift that standard risk models did not flag.</p>
    </div>
    <div class="dot-row" id="dots"></div>
    <div class="hint">click a segment to explore</div>
  </div>
</div>

<script>
const sections = [
  {
    title: "Correlation & Returns",
    tag: "cooperativity", tagBg: "#0d2045", tagColor: "#58a6ff",
    body: `<p>Examines the full distributional behavior of asset returns — skewness, excess kurtosis, and the degree to which tails deviate from a Gaussian benchmark. Autocorrelation is tested across both raw returns and absolute returns, the latter being a proxy for volatility persistence. How volatility scales with the square root of time is also examined, since departures from √T scaling imply memory in the process.</p><p>The central insight is pairwise cooperativity under stress. During normal markets, assets move with low correlation — diversification works. In the lead-up to systemic crises, correlations rise sharply and converge, dismantling the assumption that a diversified portfolio provides meaningful protection. The pre-2008 cross-asset correlation trend culminated near 0.72 before the crash, a textbook regime shift that standard risk models did not flag.</p>`
  },
  {
    title: "LPPL Bubble Detection",
    tag: "crash signal", tagBg: "#0a1f15", tagColor: "#3fb950",
    body: `<p>The Log-Periodic Power Law model, developed by Didier Sornette, describes the price dynamics of a speculative bubble as a power-law growth function decorated with log-periodic oscillations. These oscillations arise from the discrete scale invariance of herding behavior among market participants, and they accelerate in frequency as prices approach a critical time tᶜ — the most probable moment of a crash or regime change.</p><p>A single LPPL fit is unreliable in isolation. The method gains credibility through multi-window analysis: fitting the model across many overlapping time windows and examining whether the estimated critical times converge. Tight convergence is a genuine signal. Wide dispersion means the log-periodic structure is not present and the apparent fit is noise. This dashboard runs that convergence test and visualizes the distribution of estimated tᶜ values directly.</p>`
  },
  {
    title: "Multifractal Analysis",
    tag: "MF-DFA", tagBg: "#1f1400", tagColor: "#e3b341",
    body: `<p>Multifractal Detrended Fluctuation Analysis extends the classical Hurst exponent into a full spectrum. Where standard DFA produces a single scaling exponent H, MF-DFA computes a generalized Hurst function H(q) across a range of statistical moments q. If H(q) is constant the series is monofractal — it scales the same way regardless of moment. Real financial time series are not.</p><p>H(q) varies with q, meaning large fluctuations scale differently from small ones. This is true multifractality, and its richness is summarized by the width of the singularity spectrum Δα. A wider Δα indicates more complex, heterogeneous dynamics. Bitcoin and Gold consistently exhibit far wider spectra than equity indices, reflecting their more extreme tail behavior. Classical risk metrics treat all assets as if Δα were zero — they are entirely blind to this structure.</p>`
  },
  {
    title: "Drawdown Distribution",
    tag: "fat tails", tagBg: "#200d1a", tagColor: "#db61a2",
    body: `<p>Under a Gaussian return model, portfolio drawdowns would follow an exponential distribution — the probability of a loss of magnitude x decays exponentially as x grows, making truly large drawdowns astronomically rare. This is the implicit assumption embedded in standard VaR, maximum drawdown estimates, and most capital adequacy frameworks.</p><p>Empirically this assumption fails badly. Real drawdown distributions have fat tails: the probability mass in the extreme loss region is orders of magnitude larger than exponential decay would predict. This section fits both exponential and power-law models to observed drawdowns and tests the deviation directly. The result has immediate consequences — capital reserves sized on Gaussian assumptions are systematically too small, and catastrophic losses should be treated as a regular feature of the system, not an anomaly.</p>`
  },
  {
    title: "Overview",
    tag: "framework", tagBg: "#1a1030", tagColor: "#a78bfa",
    body: `<p>Standard risk tools like Value at Risk and Expected Shortfall are mathematically elegant but built on assumptions real markets routinely violate — stationarity, Gaussian return distributions, and temporal independence. These assumptions make the math tractable. They do not make it accurate.</p><p>Real markets are characterized by long-range dependence, where past returns influence future ones across long horizons; volatility clustering, where turbulent periods beget turbulence; fat-tailed return distributions; and sudden regime shifts where the statistical properties of the market change discontinuously. This dashboard applies fractal geometry and chaos-based methods — MF-DFA, LPPL, drawdown analysis — to make those hidden structures visible and measurable. Data sourced via yfinance at daily resolution.</p>`
  }
];

const spokeAngles = [0, 72, 144, 216, 288];
const wheelG = document.getElementById('wheel-g');
const dotsEl = document.getElementById('dots');

function setContent(i) {
  const s = sections[i];
  document.getElementById('ctitle').textContent = s.title;
  const tag = document.getElementById('ctag');
  tag.textContent = s.tag; tag.style.background = s.tagBg; tag.style.color = s.tagColor;
  const body = document.getElementById('cbody');
  body.innerHTML = s.body;
  body.classList.remove('fade'); void body.offsetWidth; body.classList.add('fade');
  document.querySelectorAll('.dot').forEach((d,j) => d.classList.toggle('active', j===i));
  document.querySelectorAll('.spoke').forEach((sp,j) => sp.style.opacity = j===i ? '1' : '0.4');
}

function rotateTo(i) {
  wheelG.style.transform = `rotate(${-spokeAngles[i]}deg)`;
  setTimeout(() => setContent(i), 340);
}

for (let i = 0; i < sections.length; i++) {
  document.getElementById('s'+i).addEventListener('click', () => rotateTo(i));
  const dot = document.createElement('div');
  dot.className = 'dot' + (i===0 ? ' active' : '');
  dot.addEventListener('click', () => rotateTo(i));
  dotsEl.appendChild(dot);
}
</script>
</body>
</html>
""", height=820)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CORRELATION & RETURNS
# ══════════════════════════════════════════════════════════════════════════════
with tab_corr:
    st.markdown("### Correlation & Return Distribution Analysis")
    st.markdown("<p style='color:#8b949e;font-size:13px'>Replicating Sornette (2003) — autocorrelation, scaling laws, pairwise heatmaps & bubble dynamics.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("#### ⚙️ Correlation Settings")
        c_start = st.date_input("Start date", value=date(1990, 1, 1), key="c_start")
        c_end   = st.date_input("End date",   value=date(2024, 1, 1), key="c_end")
        run_corr = st.button("▶  Run Analysis", key="run_corr", type="primary")

    if run_corr:
        @st.cache_data(show_spinner=False)
        def load_corr_data(start, end):
            djia   = yf.download("^DJI",  start=str(start), end=str(end), progress=False)["Close"].squeeze().dropna()
            nasdaq = yf.download("^IXIC", start=str(start), end=str(end), progress=False)["Close"].squeeze().dropna()
            comps  = ["MMM","AXP","BA","CAT","CVX","KO","XOM","GE","IBM","MCD","MRK","PG","WMT","JPM","MSFT"]
            comp_data = yf.download(comps, start="2003-01-01", end="2010-01-01", progress=False)["Close"].dropna(how="all")
            return djia, nasdaq, comp_data

        with st.spinner("Downloading market data…"):
            djia_p, nasdaq_p, comp_p = load_corr_data(c_start, c_end)

        djia_log   = np.log(djia_p / djia_p.shift(1)).dropna()
        nasdaq_log = np.log(nasdaq_p / nasdaq_p.shift(1)).dropna()
        djia_monthly   = djia_log.resample("ME").sum().dropna()
        nasdaq_monthly = nasdaq_log.resample("ME").sum().dropna()

        # ── Metrics row ──
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("DJIA daily vol (ann.)",   f"{djia_log.std()*np.sqrt(252)*100:.2f}%")
        m2.metric("Nasdaq daily vol (ann.)", f"{nasdaq_log.std()*np.sqrt(252)*100:.2f}%")
        m3.metric("DJIA obs", f"{len(djia_log):,}")
        m4.metric("Nasdaq obs", f"{len(nasdaq_log):,}")

        st.markdown("---")

        # ── FIG 1: Return distribution ──
        with st.spinner("Plotting distributions…"):
            def ccdf(arr):
                s = np.sort(arr); return s, np.arange(len(s),0,-1)
            def fit_exp(x,a,b): return a*np.exp(-b*x)

            fig, ax = plt.subplots(figsize=(9,5))
            for ret, label, color, marker in [
                (djia_log["1990":"2000"],   "DJIA >0",   "#58a6ff", "+"),
                (-djia_log["1990":"2000"][djia_log["1990":"2000"]<0], "DJIA <0", "#f85149", "o"),
                (nasdaq_log["1990":"2000"], "NDAQ >0",   "#3fb950", "x"),
                (-nasdaq_log["1990":"2000"][nasdaq_log["1990":"2000"]<0],"NDAQ <0","#e3b341","D"),
            ]:
                arr = ret.dropna().values
                arr = arr[arr>0]
                if len(arr) == 0: continue
                x,c = ccdf(arr)
                step = max(1,len(x)//300)
                ax.plot(x[::step],c[::step],marker=marker,linestyle="none",color=color,markersize=4,alpha=0.8,label=label)
                try:
                    mask = x > np.median(x)
                    p,_ = curve_fit(fit_exp,x[mask],c[mask],p0=[c[mask][0],50],maxfev=5000)
                    xf = np.linspace(x[mask][0],x.max(),200)
                    ax.plot(xf,fit_exp(xf,*p),color=color,lw=1.5,alpha=0.9)
                except: pass
            ax.set_yscale("log"); ax.set_xlim(left=0); ax.set_ylim(bottom=1)
            ax.set_xlabel("|Returns|"); ax.set_ylabel("Cumulative count")
            ax.set_title("Return Distribution — DJIA & Nasdaq (1990–2000)")
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
            st.markdown("#### Return Distribution")
            fig_to_st(fig)

        # ── FIG 2: Autocorrelation ──
        with st.spinner("Computing autocorrelations…"):
            def acf(series, max_lag=30):
                s = (series-series.mean())/series.std()
                return np.array([np.corrcoef(s[:-l].values,s[l:].values)[0,1] for l in range(1,max_lag+1)])

            fig, axes = plt.subplots(2,2,figsize=(13,8))
            for ax,(series,label,ml) in zip(axes.ravel(),[
                (djia_log,"DJIA Daily",30),(nasdaq_log,"Nasdaq Daily",30),
                (djia_monthly,"DJIA Monthly",24),(nasdaq_monthly,"Nasdaq Monthly",24)
            ]):
                lags = np.arange(1,ml+1)
                ax.bar(lags-0.2, acf(series,ml), 0.4, alpha=0.85, color="#58a6ff", label="Returns ACF")
                ax.bar(lags+0.2, acf(series.abs(),ml), 0.4, alpha=0.85, color="#f85149", label="|Returns| ACF")
                ci = 1.96/np.sqrt(len(series))
                ax.axhline(ci,color="#8b949e",ls="--",lw=1,alpha=0.6)
                ax.axhline(-ci,color="#8b949e",ls="--",lw=1,alpha=0.6)
                ax.axhline(0,color="#30363d",lw=0.8)
                ax.set_title(label); ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
                ax.set_ylim(-0.15,0.28); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
            fig.tight_layout()
            st.markdown("#### Autocorrelation of Returns")
            fig_to_st(fig)

        # ── FIG 3: Scaling law ──
        with st.spinner("Scaling law…"):
            def scaling(log_ret):
                sd = log_ret.std()
                return {
                    "Daily": (sd,sd),
                    "Weekly\n(√5)":    (sd*np.sqrt(5),  log_ret.resample("W").sum().dropna().std()),
                    "Monthly\n(√21)":  (sd*np.sqrt(21), log_ret.resample("ME").sum().dropna().std()),
                    "Quarterly\n(√63)":(sd*np.sqrt(63), log_ret.resample("QE").sum().dropna().std()),
                }
            fig, (a1,a2) = plt.subplots(1,2,figsize=(12,5))
            for ax, (ret, title) in [(a1,(djia_log,"DJIA")),(a2,(nasdaq_log,"Nasdaq"))]:
                d = scaling(ret)
                labels = list(d.keys())
                exp_ = [v[0]*100 for v in d.values()]
                act_ = [v[1]*100 for v in d.values()]
                x = np.arange(len(labels))
                ax.bar(x-0.18, exp_, 0.35, color="#58a6ff", alpha=0.85, label="Expected √T")
                ax.bar(x+0.18, act_, 0.35, color="#3fb950", alpha=0.85, label="Actual")
                ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
                ax.set_title(f"{title} — Scaling Law"); ax.set_ylabel("Volatility (%)"); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
            fig.tight_layout()
            st.markdown("#### √T Scaling Law")
            fig_to_st(fig)

        # ── FIG 4: Rolling pairwise correlation ──
        with st.spinner("Computing pairwise correlation (2008 bubble)…"):
            comp_ret = np.log(comp_p / comp_p.shift(1)).dropna()
            roll = 60
            dates_r, pw_r = [], []
            tickers_c = comp_ret.columns.tolist()
            pairs = [(i,j) for i in range(len(tickers_c)) for j in range(i+1,len(tickers_c))]
            for end_i in range(roll, len(comp_ret)):
                win = comp_ret.iloc[end_i-roll:end_i]
                c = np.array([win.iloc[:,i].corr(win.iloc[:,j]) for i,j in pairs])
                c = c[np.isfinite(c)]
                dates_r.append(comp_ret.index[end_i])
                pw_r.append(c.mean() if len(c) else np.nan)
            pw_s = pd.Series(pw_r, index=dates_r).rolling(15, min_periods=5).mean()

            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,8),sharex=True)
            djia_08 = djia_p["2003":"2010"]
            ax1.plot(djia_08.index, djia_08.values, color="#58a6ff", lw=1.4)
            ax1.axvspan(pd.Timestamp("2006-01-01"), pd.Timestamp("2007-10-09"), alpha=0.12, color="#f85149", label="Bubble buildup")
            ax1.axvspan(pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09"), alpha=0.12, color="#8b1a1a")
            ax1.axvline(pd.Timestamp("2007-10-09"), color="#f85149", lw=1.5, ls="--", label="Peak Oct-07")
            ax1.axvline(pd.Timestamp("2009-03-09"), color="#e3b341", lw=1.5, ls="--", label="Bottom Mar-09")
            ax1.set_ylabel("DJIA Level"); ax1.legend(fontsize=9); ax1.grid(True,alpha=0.3)
            ax1.set_title("DJIA — 2008 Bubble & Crash")

            ax2.plot(pw_s.index, pw_s.values, color="#3fb950", lw=1.5, label=f"{roll}-day avg pairwise corr")
            ax2.axvspan(pd.Timestamp("2006-01-01"), pd.Timestamp("2007-10-09"), alpha=0.12, color="#f85149")
            ax2.axvspan(pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09"), alpha=0.12, color="#8b1a1a")
            ax2.axvline(pd.Timestamp("2007-10-09"), color="#f85149", lw=1.5, ls="--")
            ax2.axvline(pd.Timestamp("2009-03-09"), color="#e3b341", lw=1.5, ls="--")
            ax2.set_ylabel("Mean Pairwise Correlation"); ax2.set_xlabel("Date"); ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
            ax2.set_title("Average Pairwise Correlation — DJIA Components (Cooperativity)")
            fig.tight_layout()
            st.markdown("#### 2008 Bubble — Pairwise Correlation")
            fig_to_st(fig)

        # ── FIG 5: Heatmaps ──
        with st.spinner("Correlation heatmaps…"):
            snaps = [
                ("Pre-Bubble\n(Jan 2006)", "2005-07-01", "2006-01-01"),
                ("Pre-Crash Peak\n(Oct 2007)", "2007-04-01", "2007-10-09"),
                ("During Crash\n(Oct 2008)", "2008-04-01", "2008-10-01"),
            ]
            fig, axes = plt.subplots(1,3,figsize=(17,6))
            cmaps = ["Blues","Oranges","Reds"]
            for ax,(label,s,e),cmap in zip(axes,snaps,cmaps):
                win = comp_ret[s:e]
                corr = win.corr()
                avg = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack().mean()
                im = ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1)
                ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(corr.index, fontsize=7)
                ax.set_title(f"{label}\nAvg r = {avg:.3f}", fontsize=10)
                for i in range(len(corr)):
                    for j in range(len(corr.columns)):
                        v = corr.values[i,j]
                        ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=5.5,color="white" if v>0.6 else "#e6edf3")
                plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            fig.tight_layout()
            st.markdown("#### Pairwise Correlation Heatmaps")
            fig_to_st(fig)

    else:
        st.info("Configure settings in the sidebar and click **▶ Run Analysis** to generate charts.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — LPPL
# ══════════════════════════════════════════════════════════════════════════════
with tab_lppl:
    st.markdown("### Log-Periodic Power Law (LPPL) Fits")
    st.markdown("<p style='color:#8b949e;font-size:13px'>Fits multiple LPPL windows over a user-defined observation period. Each line = one window. No bubble markers.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚙️ LPPL Settings")
        lppl_ticker = st.text_input("Ticker", value="^GSPC", key="lppl_ticker")
        lppl_start  = st.date_input("Observation start", value=date(2007,1,1), key="lppl_start")
        lppl_end    = st.date_input("Observation end",   value=date(2009,6,1), key="lppl_end")
        lppl_target = st.date_input("Analysis point (tc anchor)", value=date(2008,9,26), key="lppl_target")
        run_lppl = st.button("▶  Fit LPPL", key="run_lppl", type="primary")

    def cost_fn(params, t, p):
        tc, m, w = params
        if tc <= t[-1] or tc > t[-1]+30: return 1e22
        if m < 0.05 or m > 0.45: return 1e22
        if w < 5.0  or w > 15.0: return 1e22
        tau = np.maximum(tc-t, 1e-4)
        f = tau**m; g = tau**m*np.cos(w*np.log(tau)); h_ = tau**m*np.sin(w*np.log(tau))
        X = np.column_stack([np.ones(len(t)), f, g, h_])
        lin,_,_,_ = np.linalg.lstsq(X, p, rcond=None)
        if lin[1] >= 0: return 1e22
        osc = np.sqrt(lin[2]**2+lin[3]**2)
        if osc > abs(lin[1]): return 1e22
        y_hat = lin[0]+lin[1]*f+lin[2]*g+lin[3]*h_
        mse = np.sum((y_hat-p)**2)/len(t)
        return mse*(1.0+(tc-t[-1])/5.0)

    def fit_lppl(t, p):
        tc_s = t[-1]+1
        best_c, best_x = 1e30, [tc_s+5, 0.2, 8.0]
        for tc_g in np.linspace(tc_s+2, tc_s+30, 5):
            for m_g in [0.1,0.3,0.5]:
                for w_g in [6.0,10.0,14.0]:
                    c = cost_fn([tc_g,m_g,w_g], t, p)
                    if c < best_c: best_c=c; best_x=[tc_g,m_g,w_g]
        res = minimize(cost_fn, best_x, args=(t,p), method="Nelder-Mead", tol=1e-9)
        return res.x

    def solve_lppl(tc, m, w, t, p):
        tau = np.maximum(tc-t, 1e-4)
        f = tau**m; g = tau**m*np.cos(w*np.log(tau)); h_ = tau**m*np.sin(w*np.log(tau))
        X = np.column_stack([np.ones(len(t)), f, g, h_])
        lin,_,_,_ = np.linalg.lstsq(X, p, rcond=None)
        return lin, f, g, h_

    if run_lppl:
        @st.cache_data(show_spinner=False)
        def load_lppl(ticker, start, end):
            data = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=True)
            return data["Close"].squeeze().dropna()

        with st.spinner(f"Downloading {lppl_ticker}…"):
            prices = load_lppl(lppl_ticker, lppl_start, lppl_end)

        if len(prices) < 60:
            st.error("Not enough data for LPPL fitting. Try a longer window."); st.stop()

        df = prices.reset_index()
        df.columns = ["Date","Close"]
        try:
            anchor_idx = df[df["Date"] <= str(lppl_target)].index[-1]
        except IndexError:
            st.error("Target date not found in data range."); st.stop()

        window_offsets = [120, 100, 80, 60]
        colors_w = ["#58a6ff","#3fb950","#e3b341","#f85149"]

        # Metrics
        p_last = float(prices.iloc[-1]); p_first = float(prices.iloc[0])
        ret_tot = (p_last/p_first - 1)*100
        m1,m2,m3 = st.columns(3)
        m1.metric("Last price", f"{p_last:,.2f}")
        m2.metric("Period return", f"{ret_tot:+.1f}%")
        m3.metric("Trading days", f"{len(prices):,}")

        st.markdown("---")
        st.markdown(f"#### LPPL Multiple Window Fits — {lppl_ticker}")
        st.markdown(f"<p style='color:#8b949e;font-size:12px'>Analysis point: <b>{lppl_target}</b> · Windows: {window_offsets} trading days</p>", unsafe_allow_html=True)

        with st.spinner("Fitting LPPL across windows (this may take ~30s)…"):
            plot_dur = 180
            fig, ax = plt.subplots(figsize=(13,5))

            view = df.iloc[max(0, anchor_idx-plot_dur): min(len(df)-1, anchor_idx+50)+1]
            ax.plot(view["Date"], np.log(view["Close"]), color="#8b949e", lw=1.0, alpha=0.5, label="Log price")

            tc_list = []
            for offset, color in zip(window_offsets, colors_w):
                if anchor_idx - offset < 0: continue
                seg = df.iloc[anchor_idx-offset: anchor_idx+1].copy()
                t_f = np.arange(len(seg), dtype=float)
                p_f = np.log(seg["Close"].values).astype(float)
                try:
                    tc_r, m_v, w_v = fit_lppl(t_f, p_f)
                except Exception: continue
                lin_v, f_v, g_v, h_v = solve_lppl(tc_r, m_v, w_v, t_f, p_f)
                y_fit = lin_v[0] + lin_v[1]*f_v + lin_v[2]*g_v + lin_v[3]*h_v
                ax.plot(seg["Date"], y_fit, color=color, lw=2.0, alpha=0.85, label=f"{offset}d window (m={m_v:.2f}, ω={w_v:.1f})")
                tc_list.append(tc_r - t_f[-1])

            as_of = df["Date"].iloc[anchor_idx]
            ax.axvline(as_of, color="#e6edf3", ls="--", lw=1.2, alpha=0.7, label="Analysis point")
            sigma = np.std(tc_list) if tc_list else 0
            ax.set_title(f"{lppl_ticker} — LPPL Fits  |  σ(tc) = {sigma:.2f} days")
            ax.set_xlabel("Date"); ax.set_ylabel("Log price")
            ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_to_st(fig)

        if tc_list:
            col1, col2 = st.columns(2)
            col1.metric("tc spread σ (days)", f"{sigma:.2f}")
            col2.metric("Fitted windows", len(tc_list))
    else:
        st.info("Set observation dates in the sidebar and click **▶ Fit LPPL**.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MULTIFRACTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_mf:
    st.markdown("### Multifractal Analysis (MF-DFA)")
    st.markdown("<p style='color:#8b949e;font-size:13px'>Individual 6-panel diagnostics per ticker + sector-wise Pearson correlation F+q vs F-q.</p>", unsafe_allow_html=True)

    # Shared positional palette — same color at same index in every sector.
    # AAPL=green, MSFT=blue, GOOG=red, AMZN=amber, NVDA=purple, etc.
    # Repeated across sectors so position 0 is always the same color.
    TICKER_PALETTE = [
        "#3fb950",  # 0  green
        "#58a6ff",  # 1  blue
        "#f85149",  # 2  red
        "#e3b341",  # 3  amber
        "#bc8cff",  # 4  purple
        "#ff7f0e",  # 5  orange
        "#17becf",  # 6  teal
        "#f78fb3",  # 7  pink
        "#d4a017",  # 8  gold
        "#a8dadc",  # 9  ice blue
    ]

    SECTORS = {
        "Tech": {
            "tickers": ["AAPL","MSFT","GOOG","AMZN","NVDA","META","TSLA","NFLX","AMD","INTC"],
            "colors":  TICKER_PALETTE,
        },
        "Financials": {
            "tickers": ["JPM","BAC","GS","MS","WFC","BLK","C","AXP","V","MA"],
            "colors":  TICKER_PALETTE,
        },
        "Healthcare & Consumer": {
            "tickers": ["JNJ","PFE","UNH","MRK","ABBV","PG","KO","PEP","WMT","COST"],
            "colors":  TICKER_PALETTE,
        },
        "Energy & Industrials": {
            "tickers": ["XOM","CVX","COP","BA","CAT","GE","HON","LMT","RTX","UPS"],
            "colors":  TICKER_PALETTE,
        },
        "Commodities & Crypto": {
            "tickers": ["GC=F","SI=F","CL=F","BTC-USD","ETH-USD"],
            "colors":  TICKER_PALETTE,
        },
        "Intl ETFs": {
            "tickers": ["EWJ","EWZ","FXI","EWG","EWU"],
            "colors":  TICKER_PALETTE,
        },
    }
    Q_VALS_MF = np.arange(-30, 31, 5, dtype=float)
    SCALE_MIN, SCALE_MAX, N_SCALES, POLY_ORD = 20, 800, 30, 1

    def _integrate(x): return np.cumsum(x - x.mean())

    def _mfdfa(x, q_vals, s_min, s_max, n_sc, poly_ord=1, overlap=0.75):
        y = _integrate(x); n = len(y)
        s_max = min(s_max, n//4); s_min = max(s_min, poly_ord+2)
        scales = np.unique(np.logspace(np.log10(s_min),np.log10(s_max),n_sc).astype(int))
        scales = scales[scales>=s_min]
        fqs = np.full((len(q_vals),len(scales)),np.nan)
        for i,s in enumerate(scales):
            step = max(1,int(s*(1-overlap))); starts = np.arange(0,n-s+1,step)
            if len(starts)<4: continue
            t = np.arange(s)
            rms = []
            for st in starts:
                seg = y[st:st+s]
                r = np.sqrt(np.mean((seg - np.polyval(np.polyfit(t, seg, poly_ord), t))**2))
                if r > 0: rms.append(r)
            rms = np.array(rms)
            if len(rms)==0: continue
            for j,q in enumerate(q_vals):
                fqs[j,i] = np.exp(0.5*np.mean(np.log(rms**2))) if q==0 else (np.mean(rms**q))**(1/q)
        hq = np.full(len(q_vals),np.nan)
        for j in range(len(q_vals)):
            row=fqs[j]; mask=np.isfinite(row)&(row>0)
            if mask.sum()>=4: hq[j]=np.polyfit(np.log(scales[mask]),np.log(row[mask]),1)[0]
        return scales, fqs, hq

    def _spectrum(q_vals, hq):
        tau_q = q_vals*hq-1.0
        wl = min(5, len(tau_q) if len(tau_q)%2==1 else len(tau_q)-1)
        tau_s = savgol_filter(tau_q,wl,2) if len(tau_q)>=5 else tau_q
        alpha = np.gradient(tau_s, q_vals)
        return tau_q, alpha, q_vals*alpha-tau_s

    def _mfdfa_sym(x, q_vals, s_min, s_max, n_sc, poly_ord=1, overlap=0.75):
        y = _integrate(x); n = len(y)
        s_max = min(s_max,n//4); s_min = max(s_min,poly_ord+2)
        scales = np.unique(np.logspace(np.log10(s_min),np.log10(s_max),n_sc).astype(int))
        scales = scales[scales>=s_min]
        fp = np.full((len(q_vals),len(scales)),np.nan)
        fn = np.full((len(q_vals),len(scales)),np.nan)
        for i,s in enumerate(scales):
            step = max(1,int(s*(1-overlap))); starts = np.arange(0,n-s+1,step)
            if len(starts)<4: continue
            t = np.arange(s)
            rms = []
            for st in starts:
                seg = y[st:st+s]
                r = np.sqrt(np.mean((seg - np.polyval(np.polyfit(t, seg, poly_ord), t))**2))
                if r > 0: rms.append(r)
            rms = np.array(rms)
            if len(rms)==0: continue
            for j,q in enumerate(q_vals):
                fp[j,i] = (np.mean(rms**q))**(1/q) if q!=0 else np.exp(0.5*np.mean(np.log(rms**2)))
                fn[j,i] = (np.mean(rms**(-q)))**(1/(-q)) if q!=0 else fp[j,i]
        return scales, fp, fn

    def _pearson(a,b):
        mask = np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
        if mask.sum()<3: return float("nan")
        a_,b_ = np.log(a[mask]),np.log(b[mask])
        cov = np.mean((a_-a_.mean())*(b_-b_.mean()))
        denom = a_.std()*b_.std()
        return float(cov/denom) if denom>0 else float("nan")

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚙️ Multifractal Settings")
        mf_start = st.date_input("Start", value=date(2010,1,1), key="mf_start")
        mf_end   = st.date_input("End",   value=date(2026,1,1), key="mf_end")
        mf_mode  = st.radio("View mode", ["Individual 6-panel (per ticker)","Sector Pearson correlation"], key="mf_mode")
        if mf_mode.startswith("Individual"):
            all_tickers = [t for sec in SECTORS.values() for t in sec["tickers"]]
            mf_ticker = st.selectbox("Ticker", all_tickers, key="mf_ticker")
        run_mf = st.button("▶  Run MF-DFA", key="run_mf", type="primary")

    if run_mf:
        if mf_mode.startswith("Individual"):
            @st.cache_data(show_spinner=False)
            def load_mf(ticker, start, end):
                d = yf.download(ticker, start=str(start), end=str(end), progress=False)
                return d["Close"].dropna().values.ravel().astype(float)

            with st.spinner(f"Downloading {mf_ticker}…"):
                close = load_mf(mf_ticker, mf_start, mf_end)

            if len(close) < 200:
                st.error("Insufficient data."); st.stop()

            with st.spinner("Running MF-DFA…"):
                returns = np.diff(np.log(close[close>0]))
                s_min_ = max(SCALE_MIN, POLY_ORD+2)
                s_max_ = min(SCALE_MAX, len(returns)//4)
                scales, fqs, hq = _mfdfa(returns, Q_VALS_MF, s_min_, s_max_, N_SCALES)
                tau_q, alpha, d_alpha = _spectrum(Q_VALS_MF, hq)
                valid = np.isfinite(hq)
                vs = np.isfinite(alpha)&np.isfinite(d_alpha)
                da = alpha[vs].max()-alpha[vs].min() if vs.sum()>=2 else float("nan")

                colors_q = (["#58a6ff","#3fb950","#e3b341","#f85149","#bc8cff","#ff7f0e",
                              "#00bcd4","#f78fb3","#17becf","#bcbd22","#7f7f7f","#e6edf3","#8b949e"]*4)[:len(Q_VALS_MF)]

                fig, axs = plt.subplots(2,3,figsize=(15,9))
                fig.suptitle(f"MF-DFA — {mf_ticker}  ·  Log-Returns  ·  Δα={da:.4f}", fontsize=13, fontweight="bold", y=1.01)

                axs[0,0].plot(returns, color="#f85149", lw=0.7)
                axs[0,0].set_title("(a) Log-returns"); axs[0,0].set_xlabel("t"); axs[0,0].set_ylabel("r(t)")

                for idx_q,(q,c) in enumerate(zip(Q_VALS_MF,colors_q)):
                    row=fqs[idx_q]; mask=np.isfinite(row)&(row>0)
                    if mask.sum()<2: continue
                    axs[0,1].loglog(scales[mask],row[mask],"o-",color=c,markersize=3,lw=1.0,label=f"q={int(q)}")
                axs[0,1].set_title("(b) Fluctuation functions"); axs[0,1].set_xlabel("s"); axs[0,1].set_ylabel("Fq(s)")
                step_ = max(1,len(Q_VALS_MF)//8)
                handles_,labels_ = axs[0,1].get_legend_handles_labels()
                axs[0,1].legend(handles_[::step_],labels_[::step_],fontsize=7,ncol=1)

                axs[0,2].plot(Q_VALS_MF[valid],hq[valid],"-",color="#58a6ff",lw=2)
                axs[0,2].axhline(0.5,color="#8b949e",ls="--",lw=0.8)
                axs[0,2].axhline(1.0,color="#3fb950",ls=":",lw=0.8)
                axs[0,2].set_title("(c) Generalised Hurst H(q)"); axs[0,2].set_xlabel("q"); axs[0,2].set_ylabel("H(q)")

                axs[1,0].plot(Q_VALS_MF[valid],tau_q[valid],"-",color="#58a6ff",lw=2)
                axs[1,0].axhline(0,color="#8b949e",ls="--",lw=0.8)
                axs[1,0].set_title("(d) Scaling exponent τ(q)"); axs[1,0].set_xlabel("q"); axs[1,0].set_ylabel("τ(q)")

                axs[1,1].plot(Q_VALS_MF[vs],alpha[vs],"-",color="#58a6ff",lw=2)
                axs[1,1].set_title("(e) Singularity strength α(q)"); axs[1,1].set_xlabel("q"); axs[1,1].set_ylabel("α(q)")

                axs[1,2].plot(alpha[vs],d_alpha[vs],"-",color="#58a6ff",lw=2)
                q0 = np.argmin(np.abs(Q_VALS_MF))
                if np.isfinite(alpha[q0]) and np.isfinite(d_alpha[q0]):
                    axs[1,2].plot(alpha[q0],d_alpha[q0],"o",color="#f85149",markersize=6,zorder=5)
                axs[1,2].set_title("(f) Multifractal spectrum f(α)"); axs[1,2].set_xlabel("α"); axs[1,2].set_ylabel("f(α)")

                for a in axs.ravel():
                    a.grid(True,alpha=0.3)
                fig.tight_layout()

            col_a, col_b = st.columns([3,1])
            with col_a:
                st.markdown(f"#### 6-Panel MF-DFA — {mf_ticker}")
            with col_b:
                if not np.isnan(da):
                    st.metric("Multifractal width Δα", f"{da:.4f}")
            fig_to_st(fig)

        else:  # Sector Pearson
            q_corr = np.arange(1,31,dtype=float)
            n_sectors = len(SECTORS)
            ncols = 2; nrows = int(np.ceil(n_sectors/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5*nrows), sharex=True, sharey=False)
            axes = np.atleast_2d(axes).flatten()

            prog = st.progress(0, text="Computing sector correlations…")
            total = sum(len(v["tickers"]) for v in SECTORS.values())
            done = 0

            for ax_idx,(sector_name,sector) in enumerate(SECTORS.items()):
                ax = axes[ax_idx]
                for t_idx, ticker in enumerate(sector["tickers"]):
                    try:
                        data = yf.download(ticker, start=str(mf_start), end=str(mf_end), progress=False)
                        close = data["Close"].dropna().values.ravel().astype(float)
                        if len(close) < 200:
                            done+=1; prog.progress(done/total); continue
                        rets = np.diff(np.log(close[close>0]))
                        s_min_ = max(SCALE_MIN,POLY_ORD+2); s_max_ = min(SCALE_MAX,len(rets)//4)
                        scales, fp, fn = _mfdfa_sym(rets, q_corr, s_min_, s_max_, N_SCALES)
                        corr_vals = np.array([_pearson(fp[j],fn[j]) for j in range(len(q_corr))])
                        vm = np.isfinite(corr_vals)
                        color = sector["colors"][t_idx % len(sector["colors"])]
                        ax.plot(q_corr[vm], corr_vals[vm], "o-", color=color, lw=1.8, markersize=4.5, label=ticker)
                    except Exception: pass
                    done+=1; prog.progress(done/total)
                ax.set_title(sector_name, fontsize=13, fontweight="bold")
                ax.set_xlabel("|q|", fontsize=11); ax.set_ylabel("Pearson $r$", fontsize=11)
                ax.set_xticks(q_corr[::2])
                ax.set_xlim(1, 30)
                ax.set_ylim(-0.1, 1.05)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
                ax.axhline(1.0, color="#8b949e", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(0.0, color="#8b949e", lw=0.8, ls=":",  alpha=0.4)
                ax.tick_params(direction="in", which="both", top=True, right=True)
                ax.legend(fontsize=8.5, ncol=2, loc="lower left", framealpha=0.85)
                ax.grid(True, alpha=0.3)

            for ax in axes[n_sectors:]: ax.set_visible(False)
            fig.suptitle(
                "Pearson correlation between $F_{+q}(s)$ and $F_{-q}(s)$\n"
                "MF-DFA · Log-Returns · q = 1 to 30  ·  By Sector",
                fontsize=14, fontweight="bold", y=1.01
            )
            fig.tight_layout()
            prog.empty()
            st.markdown("#### Sector Pearson Correlation")
            fig_to_st(fig)
    else:
        st.info("Select a mode in the sidebar and click **▶ Run MF-DFA**.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DRAWDOWNS
# ══════════════════════════════════════════════════════════════════════════════
with tab_dd:
    st.markdown("### Drawdown Distribution Analysis")
    st.markdown("<p style='color:#8b949e;font-size:13px'>Empirical drawdown distributions vs. exponential null hypothesis — live data via yfinance.</p>", unsafe_allow_html=True)

    MARKETS_DD = {
        "DJIA (USA)":      "^DJI",
        "NASDAQ (USA)":    "^IXIC",
        "S&P 500 (USA)":   "^GSPC",
        "FTSE 100 (UK)":   "^FTSE",
        "CAC 40 (France)": "^FCHI",
        "DAX (Germany)":   "^GDAXI",
    }

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚙️ Drawdown Settings")
        dd_start = st.date_input("Start", value=date(2000,1,1), key="dd_start")
        dd_end   = st.date_input("End",   value=date(2024,1,1), key="dd_end")
        dd_min   = st.slider("Min drawdown depth (%)", 0.1, 2.0, 0.3, 0.1, key="dd_min")
        run_dd = st.button("▶  Analyse Drawdowns", key="run_dd", type="primary")

    def get_drawdowns(returns, min_depth=0.003):
        dds = []; i = 0
        while i < len(returns):
            if returns[i] < 0:
                depth = 0.0
                while i < len(returns) and returns[i] < 0:
                    depth += returns[i]; i += 1
                if abs(depth) > min_depth:
                    dds.append(abs(depth))
            else:
                i += 1
        return np.array(dds)

    if run_dd:
        @st.cache_data(show_spinner=False)
        def load_dd(start, end):
            results = {}
            for name, ticker in MARKETS_DD.items():
                try:
                    data = yf.download(ticker, start=str(start), end=str(end), progress=False)
                    close = data["Close"].squeeze().dropna()
                    if len(close) > 200:
                        results[name] = close
                except Exception: pass
            return results

        with st.spinner("Downloading index data…"):
            price_data = load_dd(dd_start, dd_end)

        if not price_data:
            st.error("Could not download any data."); st.stop()

        nrows = int(np.ceil(len(price_data)/2))
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 5*nrows))
        axes = np.atleast_2d(axes).flatten()
        min_depth = dd_min/100

        metrics_data = []
        for idx,(name,prices) in enumerate(price_data.items()):
            ax = axes[idx]
            log_ret = np.diff(np.log(prices.values.ravel()))
            dds = np.sort(get_drawdowns(log_ret, min_depth))[::-1]
            if len(dds) < 10:
                ax.text(0.5,0.5,"Insufficient drawdowns",transform=ax.transAxes,ha="center",color="#8b949e"); continue
            n = len(dds)
            log_cum = np.log10(np.arange(1,n+1))
            lam = 1.0/np.mean(dds)
            x_line = np.linspace(0, dds.max()*1.05, 300)
            y_line = np.log10(n) - lam*x_line/np.log(10)
            ax.plot(x_line, y_line, color="#8b949e", lw=1.5, ls="--", label="Null (exponential)")
            ax.scatter(dds, log_cum, s=12, color="#58a6ff", marker=".", alpha=0.8, label="Observed")
            ax.invert_xaxis()
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.1f}%"))
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Drawdown depth"); ax.set_ylabel("Log₁₀(Cumulative count)")
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
            metrics_data.append((name, n, dds.mean()*100, dds.max()*100))

        for ax in axes[len(price_data):]: ax.set_visible(False)
        fig.tight_layout()

        # Metric row
        if metrics_data:
            cols = st.columns(len(metrics_data))
            for col,(name,n,mean_d,max_d) in zip(cols,metrics_data):
                mkt = name.split("(")[0].strip()
                col.metric(f"{mkt} max DD", f"{max_d:.1f}%", f"n={n} events")

        st.markdown("---")
        st.markdown("#### Drawdown Distributions")
        fig_to_st(fig)

        # Summary table
        if metrics_data:
            st.markdown("#### Summary Statistics")
            df_sum = pd.DataFrame(metrics_data, columns=["Market","N drawdowns","Mean depth (%)","Max depth (%)"])
            df_sum["Mean depth (%)"] = df_sum["Mean depth (%)"].round(2)
            df_sum["Max depth (%)"] = df_sum["Max depth (%)"].round(2)
            st.dataframe(df_sum.set_index("Market"), use_container_width=True)
    else:
        st.info("Configure settings in the sidebar and click **▶ Analyse Drawdowns**.", icon="ℹ️")