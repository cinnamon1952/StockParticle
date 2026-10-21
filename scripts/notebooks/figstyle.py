"""Shared figure style for every figure in the paper (import this, do not restyle locally).

Widths are real journal widths (Elsevier: 90 mm single, 190 mm double column), so a font size of 8 pt
here IS 8 pt on the printed page. Colours: Okabe-Ito (colour-blind safe); markers differ per group so the
groups stay distinguishable in grayscale. American spelling in all figure text.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG_W_1COL = 3.54      # inches (90 mm)
FIG_W_2COL = 7.48      # inches (190 mm)
FONT_PT = 8

GROUP_ORDER = ["Tech", "Financials", "Health/Cons", "Energy/Ind", "Commodity", "Crypto", "Intl ETF"]
GROUP_COLORS = {"Tech": "#0072B2", "Financials": "#56B4E9", "Health/Cons": "#009E73", "Energy/Ind": "#D55E00",
                "Commodity": "#E69F00", "Crypto": "#CC79A7", "Intl ETF": "#4D4D4D"}
GROUP_MARKERS = {"Tech": "o", "Financials": "s", "Health/Cons": "^", "Energy/Ind": "D",
                 "Commodity": "v", "Crypto": "P", "Intl ETF": "X"}
# surrogate ensembles (Fig. 3 / Fig. A.x): fixed colors and markers, also gray-safe
SURR_STYLE = {"shuffle": dict(c="#D55E00", marker="s"), "iaaft": dict(c="#009E73", marker="D"),
              "phase": dict(c="#7F7F7F", marker="^")}


def apply():
    mpl.rcParams.update({
        "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
        "legend.title_fontsize": FONT_PT, "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.unicode_minus": True,                      # true minus sign on all axes
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,           # embed TrueType: text stays text in the PDF
    })


def group_handles(with_marker=True):
    return [Line2D([], [], marker=GROUP_MARKERS[g] if with_marker else "o", ls="", mfc=GROUP_COLORS[g],
                   mec="k", mew=0.4, ms=5, label=g) for g in GROUP_ORDER]


def panel_label(ax, text, x=-0.02, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=FONT_PT, fontweight="bold", va="bottom", ha="right")


def save(fig, name, outdir):
    """Vector PDF + 300 dpi PNG, bbox_inches='tight'. `name` has no extension."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
