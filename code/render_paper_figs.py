"""Renders Figures 2 to 6 of the manuscript at their printed size (5.1 inches wide, the text width of the
Springer Nature template), so font sizes in the PDF are the sizes on the page. Reads the result tables only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

ROOT = Path(__file__).resolve().parent.parent      # repository root
B = ROOT / "results"
OUT = ROOT / "figures_paper"
OUT.mkdir(exist_ok=True)
W = 5.1                                     # text width of the Springer Nature template (372 pt) in inches
SPINE, LEG = 1.0, 0.6                       # legend outline thinner than spines
BLUE, ORANGE, GREY, BLACK = "#2a78d6", "#eb6834", "#8a8a8a", "#000000"
MODELS = [("resnet50_swsl", "ResNet-50 (SWSL)", "layer3"), ("cornet_s", "CORnet-S", "V4")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.linewidth": SPINE, "xtick.major.width": SPINE,
    "ytick.major.width": SPINE, "xtick.major.size": 3.5, "ytick.major.size": 3.5, "legend.frameon": True,
    "legend.fancybox": False, "legend.edgecolor": "black", "legend.framealpha": 1.0, "pdf.fonttype": 42,
    "savefig.dpi": 300, "lines.linewidth": 1.3, "errorbar.capsize": 3})


def axes_style(ax, lo, hi, step, dec):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(SPINE)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(np.round(np.arange(lo, hi + step / 2, step), 10)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{dec}f}"))
    ax.tick_params(direction="out")


def legend(obj, *args, **kw):
    lg = obj.legend(*args, **kw)
    lg.get_frame().set_linewidth(LEG)
    return lg


PENDING = []          # panel letters, placed at save time on the centre line of each y-axis title


def letter(ax, s, x=None):
    PENDING.append((ax, s))


def place_letters(fig):
    """Put each panel letter in the same column as its panel's y-axis title: the letter's horizontal centre
    is the measured horizontal centre of the rendered title, and it sits just above the panel."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax, s in PENDING:
        if ax.figure is not fig:
            continue
        bb = ax.yaxis.label.get_window_extent(renderer=fig.canvas.get_renderer())
        xc = inv.transform(((bb.x0 + bb.x1) / 2, 0))[0]
        top = inv.transform(ax.transAxes.transform((0, 1)))[1]
        fig.text(xc, top + 0.012, s, ha="center", va="bottom", fontsize=12, fontweight="bold")
    PENDING[:] = [p for p in PENDING if p[0].figure is not fig]


def save(fig, name):
    place_letters(fig)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def check(lo, hi, vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    assert vals.min() >= lo and vals.max() <= hi, (lo, hi, vals.min(), vals.max())


# ----------------------------------------------------------------------------- Figure 2
acc = pd.read_csv(B / "stage9/figure1_accuracies.csv")
conds = ["intact", "configural", "local", "random (damage-matched)"]
xl = ["Intact", "Configural lesion", "Local lesion", "Random lesions"]
fig, axes = plt.subplots(2, 1, figsize=(W, 5.6))
LX = -0.085
for p, (ax, (m, lab, site)) in enumerate(zip(axes, MODELS)):
    g = acc[acc.model == m]
    for meas, mfc, dx, ml in [("grid", BLACK, -0.13, "Grid control"), ("scrambled", "white", 0.13, "Scrambled")]:
        r = g[g.measure == meas].set_index("condition").reindex(conds)
        ax.errorbar(np.arange(4) + dx, r.value, yerr=[r.value - r.lo, r.hi - r.value], fmt="o", ms=5, color=BLACK,
                    mfc=mfc, mec=BLACK, mew=1.2, lw=1.2, label=ml if p == 0 else None)
        check(0, 0.5, list(r.lo) + list(r.hi))
    ax.axhline(1 / 16, color=GREY, ls=":", lw=1.2, label="Chance (1/16)" if p == 0 else None)
    ax.axhline(2 / 16, color=GREY, ls="--", lw=1.2, label="Floor (2/16)" if p == 0 else None)
    axes_style(ax, 0, 0.5, 0.1, 1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xl)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("Accuracy (16-way)")
    ax.yaxis.set_label_coords(LX - 0.02, 0.5)
    letter(ax, "ab"[p], LX - 0.02)
h, l = axes[0].get_legend_handles_labels()
legend(fig, h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
fig.subplots_adjust(hspace=0.32, bottom=0.11)
save(fig, "fig2_lesions")

# ----------------------------------------------------------------------------- Figure 3
dose = pd.read_csv(B / "stage5/dose_response.csv")
fig, axes = plt.subplots(2, 1, figsize=(W, 5.6))
for p, (ax, (m, lab, site)) in enumerate(zip(axes, MODELS)):
    d = dose[dose.model == m].sort_values("fraction")
    x = np.arange(len(d))
    ax.fill_between(x, d.dCR_random_min, d.dCR_random_max, color=GREY, alpha=0.3, lw=0,
                    label="Random range" if p == 0 else None)
    ax.plot(x, d.dCR_random_mean, color=GREY, marker="D", ms=4, lw=1.3, label="Random mean" if p == 0 else None)
    ok = ~d.configural_floor.astype(bool).to_numpy()
    y = d.dCR_configural.to_numpy()
    ax.errorbar(x[ok], y[ok], yerr=[y[ok] - d.dCR_configural_lo.to_numpy()[ok], d.dCR_configural_hi.to_numpy()[ok] - y[ok]],
                color=BLUE, marker="o", ms=5, lw=1.3, label="Configural channels" if p == 0 else None)
    for xf in x[~ok]:
        ax.text(xf, 0.08, "floor", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.axhline(0, color=BLACK, lw=0.7)
    check(-0.75, 0.75, list(d.dCR_random_min) + list(d.dCR_random_max) + list(d.dCR_configural_lo) + list(d.dCR_configural_hi))
    axes_style(ax, -0.75, 0.75, 0.25, 2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{100 * f:g}" for f in d.fraction])
    ax.set_xlim(-0.5, len(d) - 0.5)
    ax.set_xlabel(f"Channels lesioned at {site} (%)")
    ax.set_ylabel("Change in\nconfigural reliance")
    ax.yaxis.set_label_coords(LX - 0.02, 0.5)
    letter(ax, "ab"[p], LX - 0.02)
h, l = axes[0].get_legend_handles_labels()
legend(fig, h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(hspace=0.45, bottom=0.12)
save(fig, "fig3_dose")

# ----------------------------------------------------------------------------- Figure 4
rm = pd.read_csv(B / "stage6b/readout_measures.csv")
lev = pd.read_csv(B / "stage9/figure3_retraining_levels.csv")
sel = [("none", "intact", "Intact"), ("zero", "configural", "Configural lesion,\nzero ablation"),
       ("mean", "configural", "Configural lesion,\nmean ablation")]
cols = {"resnet50_swsl": BLUE, "cornet_s": ORANGE}
fig = plt.figure(figsize=(W, 6.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.55)
ax = fig.add_subplot(gs[0, :])
for i, (m, lab, _) in enumerate(MODELS):
    for meas, filled, mk, ml in [("probe", True, "o", "linear probe"), ("accuracy", False, "s", "network output")]:
        v, lo, hi = [], [], []
        for abl, cond, _ in sel:
            r = rm[(rm.model == m) & (rm.ablation == abl) & (rm.condition == cond) & (rm.measure == meas)].iloc[0]
            v.append(r.value); lo.append(r.lo); hi.append(r.hi)
        v, lo, hi = map(np.array, (v, lo, hi))
        check(0, 1, list(lo) + list(hi))
        ax.errorbar(np.arange(3) + (i - 0.5) * 0.3, v, yerr=[v - lo, hi - v], fmt=mk, ms=5.5, lw=1.2, color=cols[m],
                    mfc=cols[m] if filled else "white", mec=cols[m], mew=1.3, label=f"{lab}, {ml}")
axes_style(ax, 0, 1, 0.2, 1)
ax.set_xticks(range(3))
ax.set_xticklabels([s_[2] for s_ in sel])
ax.set_xlim(-0.55, 2.55)
ax.set_ylabel("Grid-control\naccuracy (16-way)")
LXa = -0.085
ax.yaxis.set_label_coords(LXa, 0.5)
letter(ax, "a", LXa)
legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handletextpad=0.3, columnspacing=1.0)
mk = {"no lesion": ("o", "white", BLACK, "No lesion"), "configural": ("o", BLUE, BLUE, "Configural lesion"),
      "random (mean of 3 orders)": ("D", GREY, GREY, "Random lesions")}
regimes = [("readout", "Readout"), ("downstream", "Downstream")]
for p, (m, lab, site) in enumerate(MODELS):
    ax = fig.add_subplot(gs[1, p])
    g = lev[lev.model == m]
    for k, (les, (sym, mfc, col, ll)) in enumerate(mk.items()):
        r = g[g.lesion == les].set_index("regime").reindex([r_ for r_, _ in regimes])
        x = np.arange(2) + (k - 1) * 0.22
        check(0, 0.8, list(r.lo) + list(r.hi))
        ax.errorbar(x, r.CR, yerr=[r.CR - r.lo, r.hi - r.CR], fmt=sym, ms=5.5, lw=1.2, color=col, mfc=mfc, mec=col,
                    mew=1.3, label=ll if p == 0 else None)
    axes_style(ax, 0, 0.8, 0.2, 1)
    ax.set_xticks(range(2))
    ax.set_xticklabels([r_[1] for r_ in regimes])
    ax.set_xlim(-0.42, 1.42)
    ax.set_xlabel("Layers retrained")
    ax.set_ylabel("Configural reliance\nafter retraining")
    LXb = -0.2
    ax.yaxis.set_label_coords(LXb, 0.5)
    letter(ax, "bc"[p], LXb)
h, l = fig.axes[1].get_legend_handles_labels()
legend(fig, h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), handletextpad=0.3, columnspacing=1.0)
save(fig, "fig4_readout")

# ----------------------------------------------------------------------------- Figure 5
gi = pd.read_csv(B / "stage4a/group_interactions.csv")
gi = gi[gi.metric.isna()]
cs = pd.read_csv(B / "stage4b/components_summary.csv")
c5 = [("intact", "Intact"), ("configural lesion", "Configural lesion"), ("local lesion", "Local lesion"),
      ("random (damage-matched)", "Random lesions")]
fig = plt.figure(figsize=(W, 7.3))
gs = fig.add_gridspec(3, 1, hspace=0.42)
ax = fig.add_subplot(gs[0])
groups = [("configural", "Configural channels", BLUE), ("local", "Local channels", ORANGE), ("other", "Other channels", GREY)]
w = 0.22
for j, (gname, gl, col) in enumerate(groups):
    for i, (m, lab, site) in enumerate(MODELS):
        r = gi[(gi.model == m) & (gi.group == gname)].iloc[0]
        check(0, 0.5, [r.lo, r.hi])
        x = i + (j - 1) * w
        ax.bar(x, r["mean"], w * 0.92, color=col, edgecolor=BLACK, lw=0.8, label=gl if i == 0 else None)
        ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt="none", color=BLACK, lw=1.0)
bag = gi[gi.model == "bagnet33"].iloc[0]
ax.axhline(bag["mean"], color=BLACK, ls="--", lw=1.2, label="BagNet-33 channels")
axes_style(ax, 0, 0.5, 0.1, 1)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"{lab}, {site}" for _, lab, site in MODELS])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("Non-additivity of\nchannel response")
ax.yaxis.set_label_coords(LX - 0.02, 0.5)
letter(ax, "a", LX - 0.02)
legend(ax, loc="upper center", ncol=2, fontsize=9)
for p, (m, lab, site) in enumerate(MODELS):
    ax = fig.add_subplot(gs[1 + p])
    g = cs[cs.model == m].set_index("condition")
    lo_, hi_, st, dec = {"resnet50_swsl": (0.0, 2.0, 0.5, 1), "cornet_s": (-0.25, 1.0, 0.25, 2)}[m]
    for k, (comp, cl, col) in enumerate([("DA", "Position-coded parts", BLUE), ("DI", "Interactions between parts", ORANGE)]):
        x = np.arange(4) + (k - 0.5) * 0.36
        v = np.array([g.loc[c, comp] for c, _ in c5])
        lo = np.array([g.loc[c, f"{comp}_lo"] for c, _ in c5])
        hi = np.array([g.loc[c, f"{comp}_hi"] for c, _ in c5])
        check(lo_, hi_, list(lo) + list(hi))
        ax.bar(x, v, 0.33, color=col, edgecolor=BLACK, lw=0.8, label=cl if p == 0 else None)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", color=BLACK, lw=1.0)
    ax.axhline(0, color=BLACK, lw=0.7)
    axes_style(ax, lo_, hi_, st, dec)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[1] for c in c5])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("Scramble effect on\nevidence (logit)")
    ax.yaxis.set_label_coords(LX - 0.02, 0.5)
    letter(ax, "bc"[p], LX - 0.02)
    if p == 0:
        legend(ax, loc="upper right")
save(fig, "fig5_interactions")

# ----------------------------------------------------------------------------- Figure 6 (follow-up lesions)
fu = pd.read_csv(B / "followup/followup_accuracies.csv")
fig, axes = plt.subplots(2, 1, figsize=(W, 5.6))
for p, (ax, (m, sets)) in enumerate(zip(axes, [("resnet50_swsl", "matched cost"), ("resnet50", "planned")])):
    g = fu[(fu.model == m) & (fu.sets == sets)]
    for meas, mfc, dx, ml in [("grid", BLACK, -0.13, "Grid control"), ("scrambled", "white", 0.13, "Scrambled")]:
        r = g[g.measure == meas].set_index("condition").reindex(conds)
        ax.errorbar(np.arange(4) + dx, r.value, yerr=[r.value - r.lo, r.hi - r.value], fmt="o", ms=5, color=BLACK,
                    mfc=mfc, mec=BLACK, mew=1.2, lw=1.2, label=ml if p == 0 else None)
        check(0, 0.5, list(r.lo) + list(r.hi))
    ax.axhline(1 / 16, color=GREY, ls=":", lw=1.2, label="Chance (1/16)" if p == 0 else None)
    ax.axhline(2 / 16, color=GREY, ls="--", lw=1.2, label="Floor (2/16)" if p == 0 else None)
    axes_style(ax, 0, 0.5, 0.1, 1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xl)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("Accuracy (16-way)")
    ax.yaxis.set_label_coords(LX - 0.02, 0.5)
    letter(ax, "ab"[p], LX - 0.02)
h, l = axes[0].get_legend_handles_labels()
legend(fig, h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
fig.subplots_adjust(hspace=0.32, bottom=0.11)
save(fig, "fig6_followup")
print("figures written")
