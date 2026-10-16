"""
STAGE 1b: Gap-width sensitivity of the grid-scramble validation
CNN visual agnosia project

Why this exists
  In Stage 1 v2 (textured COCO objects), the local-only control BagNet-33 lost 0.054
  [0.041, 0.068] under grid scrambling, just outside the pre-set +-0.05 equivalence bound,
  while BagNet-17 lost 0.045 and BagNet-9 lost 0.010 (ResNet-50: 0.293). The drop grows
  with BagNet receptive field (9 -> 17 -> 33 px), which points to one mechanism: with a
  6 px gap, 17 and 33 px patches straddle neighbouring pieces, and scrambling changes what
  sits on either side of each gap. That is a genuine local change at the scale of those
  receptive fields.

  If this account is right, widening the gap should shrink the BagNet-33 drop (fewer
  patches see two pieces), while ResNet-50's excess drop over BagNet-33 (the part not
  explained by local content) should stay large.

Decision rule (fixed before running)
  1. Primary gap = the smallest gap at which BagNet-33 passes the floor check and its drop
     CI lies inside +-0.05, and ResNet-50's drop CI excludes zero. Later stages use that gap.
  2. If no gap qualifies, keep the 6 px gap and define the scramble effect for every model
     as its excess drop over BagNet-33 (paired per source), reported with the raw drop.

Inputs: Stage 1 outputs on Drive (stimulus archive, manifest, COCO masks, agnosia_common.py).
Outputs: Drive/results/stage1b/*.csv|json, Drive/figures/stage1b/*.pdf|png
Single cell: paste and run (GPU runtime). About 15 to 20 minutes on an L4.
"""

# =============================================================================
# 0. Setup
# =============================================================================
import os
import sys
import json
import time
import shutil
import subprocess
import importlib
from pathlib import Path

IN_COLAB = os.path.exists("/content") and "AGNOSIA_TEST" not in os.environ
if IN_COLAB:
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/wielandbrendel/bag-of-local-features-models.git"], check=True)

COMMON_DIR = Path(os.environ.get("AGNOSIA_ROOT", "/content/drive/MyDrive/AGNOSIA_CNN"))
assert (COMMON_DIR / "agnosia_common.py").exists(), "agnosia_common.py not found on Drive: run Stage 1 first."
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
import agnosia_common as C
C = importlib.reload(C)

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RUN_MODELS = os.environ.get("AGNOSIA_NO_MODELS") is None

CONFIG = dict(
    GAPS=[6, 12, 18, 24],
    GRID=3,
    N_SCRAMBLE=3,
    MIN_PIECE_PX=40,
    MIN_MANIP_STRENGTH=0.10,
    JPEG_QUALITY=95,
    MODELS=["bagnet9", "bagnet17", "bagnet33", "resnet50"],
    LOCAL_REF="bagnet33",
    HEADROOM_MODEL="resnet50",
    N_BOOT=10000,
    EQ_BOUND=0.05,
    FLOOR_MULT=2.0,
    BATCH=128,
)
FORCE_PREDICTIONS = False

C.ensure_dirs()
C.set_seed()
H = W = C.IMG_SIZE
RES = C.RESULTS_DIR / "stage1b"
FIGS = C.FIG_DIR / "stage1b"
WORK = C.STIM_LOCAL / "stage1b"
for d in [RES, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
MODEL_LABEL = {"bagnet9": "BagNet-9", "bagnet17": "BagNet-17", "bagnet33": "BagNet-33", "resnet50": "ResNet-50"}
print(f"Stage 1b | gaps {CONFIG['GAPS']} | models {RUN_MODELS}")


# =============================================================================
# 1. Rebuild textured layers from Stage 1 outputs
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("stage1")
S1 = C.STIM_LOCAL / "stage1"
man1 = pd.read_csv(C.STIM_DRIVE / "stage1_manifest.csv", low_memory=False)
intact = man1[(man1.battery == "configural_tex") & (man1.condition == "intact")].set_index("source_id")
npz = np.load(S1 / "masks_coco.npz")
mask_by_id = dict(zip(npz["source_ids"].tolist(), npz["masks"]))
sources = [sid for sid in intact.index if sid in mask_by_id]
print(f"  textured sources: {len(sources)}")


def bg_layers():
    L = np.zeros((H, W, 4), np.uint8)
    L[..., 1:] = 255
    return L


def mask_of(L):
    return L[..., 0] > 0


def clear(view):
    view[..., 0] = 0
    view[..., 1:] = 255


def paste_over(dst, src):
    m = mask_of(src)
    dst[m] = src[m]
    return dst


def bbox(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def load_layers(sid):
    mask = mask_by_id[sid]
    rgb = np.asarray(Image.open(C.STIM_LOCAL / intact.loc[sid, "path"]).convert("RGB"))
    L = bg_layers()
    L[mask, 0] = 1
    L[mask, 1:] = rgb[mask]
    return L


def grid_geometry(mask):
    G = CONFIG["GRID"]
    r0, r1, c0, c1 = bbox(mask)
    h, w = r1 - r0, c1 - c0
    ch, cw = min(int(np.ceil(h / G)), H // G), min(int(np.ceil(w / G)), W // G)
    R0 = int(np.clip(r0 - (G * ch - h) // 2, 0, H - G * ch))
    C0 = int(np.clip(c0 - (G * cw - w) // 2, 0, W - G * cw))
    return R0, C0, ch, cw


def grid_control(L, gap):
    G, half = CONFIG["GRID"], gap // 2
    geom = grid_geometry(mask_of(L))
    R0, C0, ch, cw = geom
    ctrl = bg_layers()
    ctrl[R0:R0 + G * ch, C0:C0 + G * cw] = L[R0:R0 + G * ch, C0:C0 + G * cw]
    for i in range(G):
        for j in range(G):
            cell = ctrl[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw]
            clear(cell[:half]); clear(cell[-half:]); clear(cell[:, :half]); clear(cell[:, -half:])
    return ctrl, geom


def grid_permutation(ctrl, geom, seed):
    G = CONFIG["GRID"]
    R0, C0, ch, cw = geom
    cm = mask_of(ctrl)
    cells = [cm[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw] for i in range(G) for j in range(G)]
    nonempty = [k for k, c in enumerate(cells) if c.sum() >= CONFIG["MIN_PIECE_PX"]]
    if len(nonempty) < 2:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(1000):
        perm = rng.permutation(G * G)
        if not all(perm[k] != k for k in nonempty):
            continue
        out = np.zeros_like(cm)
        for k, cell in enumerate(cells):
            i, j = divmod(perm[k], G)
            out[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw] |= cell
        strength = 1 - iou(cm, out)
        if best is None or strength > best[1]:
            best = (perm, strength)
        if strength >= CONFIG["MIN_MANIP_STRENGTH"]:
            break
    return best


def apply_permutation(ctrl, geom, perm):
    G = CONFIG["GRID"]
    R0, C0, ch, cw = geom
    cells = [ctrl[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw].copy()
             for i in range(G) for j in range(G)]
    out = bg_layers()
    for k, cell in enumerate(cells):
        i, j = divmod(perm[k], G)
        paste_over(out[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw], cell)
    return out


def save_jpg(L, rel):
    p = C.STIM_LOCAL / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(L[..., 1:])).save(p, quality=CONFIG["JPEG_QUALITY"], subsampling=0)


# =============================================================================
# 2. Stimuli for every gap (local disk only; regenerated on each run)
# =============================================================================
print("\n[1] Stimuli")
t0 = time.time()
if WORK.exists():
    shutil.rmtree(WORK)
rows = []
for sid in sources:
    L = load_layers(sid)
    base = dict(source_id=sid, category=intact.loc[sid, "category"], split=intact.loc[sid, "split"])
    # the shared intact image is re-encoded once so every condition has identical compression history
    rel = f"stage1b/intact/{sid}.jpg"
    save_jpg(L, rel)
    rows.append(dict(path=rel, gap=0, condition="intact", variant=0, **base))
    for gap in CONFIG["GAPS"]:
        ctrl, geom = grid_control(L, gap)
        cell_px = int(min(geom[2], geom[3]))
        rel = f"stage1b/gap{gap}/cutG_intact/{sid}.jpg"
        save_jpg(ctrl, rel)
        rows.append(dict(path=rel, gap=gap, condition="cutG_intact", variant=0, cell_px=cell_px, **base))
        for v in range(CONFIG["N_SCRAMBLE"]):
            r = grid_permutation(ctrl, geom, C.derive_seed("scramble", sid, v))
            if r is None:
                continue
            perm, strength = r
            rel = f"stage1b/gap{gap}/scrambled/{sid}__v{v}.jpg"
            save_jpg(apply_permutation(ctrl, geom, perm), rel)
            rows.append(dict(path=rel, gap=gap, condition="scrambled", variant=v, cell_px=cell_px,
                             manip_strength=strength, **base))
man = pd.DataFrame(rows)
man.to_csv(RES / "manifest_stage1b.csv", index=False)
print(f"  {len(man)} images in {time.time() - t0:0.0f} s")
print(man.groupby(["gap", "condition"]).size().to_string())


# =============================================================================
# 3. Predictions
# =============================================================================
def run_predictions():
    import torch
    device = C.get_device()
    paths = [str(C.STIM_LOCAL / p) for p in man.path]
    preds = {}
    for name in CONFIG["MODELS"]:
        out = RES / f"predictions_{name}.csv"
        if out.exists() and not FORCE_PREDICTIONS:
            df = pd.read_csv(out)
            if df.path.tolist() == man.path.tolist():
                preds[name] = df
                print(f"  {name}: cached")
                continue
        t0 = time.time()
        model = C.load_model(name, device)
        p16 = C.predict_probs16(model, paths, device, batch_size=CONFIG["BATCH"])
        del model
        torch.cuda.empty_cache()
        df = man[["path", "gap", "condition", "variant", "source_id", "category", "split"]].copy()
        y = df.category.map(C.CAT_TO_IDX).to_numpy()
        df["correct"] = (p16.argmax(1) == y).astype(int)
        df["p_true"] = p16[np.arange(len(y)), y]
        df.to_csv(out, index=False)
        preds[name] = df
        print(f"  {name}: {time.time() - t0:0.0f} s")
    return preds


# =============================================================================
# 4. Analysis
# =============================================================================
def per_source_drops(df):
    """Per source and gap: control accuracy, scrambled accuracy (mean over variants), drop."""
    d = df.merge(man[["path", "manip_strength"]], on="path", how="left")
    d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_MANIP_STRENGTH"]))]
    intact_acc = d[d.condition == "intact"].set_index("source_id")["correct"]
    out = []
    for gap in CONFIG["GAPS"]:
        g = d[d.gap == gap]
        ctrl = g[g.condition == "cutG_intact"].set_index("source_id")["correct"]
        scr = g[g.condition == "scrambled"].groupby("source_id")["correct"].mean()
        both = ctrl.index.intersection(scr.index)
        out.append(pd.DataFrame(dict(gap=gap, source_id=both, intact=intact_acc.reindex(both).to_numpy(),
                                     ctrl=ctrl.loc[both].to_numpy(), scr=scr.loc[both].to_numpy())))
    out = pd.concat(out, ignore_index=True)
    out["drop"] = out.ctrl - out.scr
    out["cut_cost"] = out.intact - out.ctrl
    return out


def analyse(preds):
    rng = np.random.default_rng(C.derive_seed("stage1b"))
    floor = CONFIG["FLOOR_MULT"] * C.CHANCE_16
    ps = {m: per_source_drops(df) for m, df in preds.items()}
    ref = ps[CONFIG["LOCAL_REF"]].set_index(["gap", "source_id"])
    rows = []
    for m, d in ps.items():
        for gap, g in d.groupby("gap"):
            cm, clo, chi = C.bootstrap_mean_ci(g.ctrl, CONFIG["N_BOOT"], rng=rng)
            dm, dlo, dhi = C.bootstrap_mean_ci(g["drop"], CONFIG["N_BOOT"], rng=rng)
            km, klo, khi = C.bootstrap_mean_ci(g.cut_cost, CONFIG["N_BOOT"], rng=rng)
            gi = g.set_index("source_id")
            r = ref.loc[gap].reindex(gi.index)
            ex = (gi["drop"] - r["drop"]).dropna()
            em, elo, ehi = C.bootstrap_mean_ci(ex, CONFIG["N_BOOT"], rng=rng)
            floor_ok = bool(clo > floor)
            rows.append(dict(model=m, gap=gap, n_sources=len(g),
                             ctrl_acc=cm, ctrl_lo=clo, ctrl_hi=chi,
                             drop=dm, drop_lo=dlo, drop_hi=dhi,
                             cut_cost=km, cut_cost_lo=klo, cut_cost_hi=khi,
                             excess_over_local=em, excess_lo=elo, excess_hi=ehi,
                             floor_ok=floor_ok, drop_sig=bool(dlo > 0 or dhi < 0),
                             decision=C.equivalence_decision(dlo, dhi, dm, CONFIG["EQ_BOUND"]) if floor_ok else "floor"))
    return pd.DataFrame(rows), ps


def decide(summary):
    loc = summary[summary.model == CONFIG["LOCAL_REF"]].set_index("gap")
    hr = summary[summary.model == CONFIG["HEADROOM_MODEL"]].set_index("gap")
    chosen = None
    for gap in CONFIG["GAPS"]:
        if loc.loc[gap, "decision"] == "equivalent" and hr.loc[gap, "drop_sig"]:
            chosen = gap
            break
    if chosen is not None:
        return dict(status="PASS", rule="equivalence at smallest qualifying gap", primary_gap=int(chosen),
                    primary_metric="raw drop (local-only control equivalent at this gap)")
    return dict(status="PASS_WITH_LOCAL_BASELINE", rule="no gap gave equivalence",
                primary_gap=CONFIG["GAPS"][0],
                primary_metric=f"excess drop over {MODEL_LABEL[CONFIG['LOCAL_REF']]} (paired per source)")


def fig_gap(summary):
    """Drop (a) and excess drop over BagNet-33 (b) against gap width."""
    C.set_pub_style()
    markers = {"bagnet9": "o", "bagnet17": "s", "bagnet33": "D", "resnet50": "^"}
    colours = {"bagnet9": "#9ecae1", "bagnet17": "#4292c6", "bagnet33": "#08519c", "resnet50": "#d94801"}
    gaps = CONFIG["GAPS"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    off = {m: (k - 1.5) * 0.35 for k, m in enumerate(CONFIG["MODELS"])}
    for p, (ax, col, lo, hi, ylab) in enumerate([
            (axes[0], "drop", "drop_lo", "drop_hi", "Scrambling drop (accuracy)"),
            (axes[1], "excess_over_local", "excess_lo", "excess_hi", "Excess drop over BagNet-33")]):
        for m in CONFIG["MODELS"]:
            if p == 1 and m == CONFIG["LOCAL_REF"]:
                continue
            g = summary[summary.model == m].set_index("gap").reindex(gaps)
            x = np.array(gaps) + off[m]
            ax.errorbar(x, g[col], yerr=[g[col] - g[lo], g[hi] - g[col]], fmt=markers[m] + "-", ms=8,
                        capsize=4, lw=1.5, color=colours[m], label=MODEL_LABEL[m])
        ax.axhline(0, color="0.3", lw=1.0)
        if p == 0:
            ax.axhspan(-CONFIG["EQ_BOUND"], CONFIG["EQ_BOUND"], color="0.85", zorder=0)
        vals = np.concatenate([summary[lo].to_numpy(), summary[hi].to_numpy()]) if p == 0 else \
            np.concatenate([summary[summary.model != CONFIG["LOCAL_REF"]][lo].to_numpy(),
                            summary[summary.model != CONFIG["LOCAL_REF"]][hi].to_numpy()])
        ymin = min(-0.1, np.floor(np.nanmin(vals) / 0.1) * 0.1)
        ymax = max(0.1, np.ceil(np.nanmax(vals) / 0.1) * 0.1)
        C.style_axes(ax, xlim=(0, 30), xstep=6, ylim=(ymin, ymax), ystep=0.1, decimals=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v + 0.0:0.1f}" if abs(v) > 1e-9 else "0.0"))
        ax.set_xlabel("Gap width (px)")
        ax.set_ylabel(ylab)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.text(-0.12, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=20, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    C.save_figure(fig, FIGS / "stage1b_gap_sensitivity")
    plt.close(fig)


if RUN_MODELS:
    print("\n[2] Predictions")
    preds = run_predictions()
    print("\n[3] Analysis")
    summary, ps = analyse(preds)
    summary.to_csv(RES / "gap_summary.csv", index=False)
    for m, d in ps.items():
        d.to_csv(RES / f"per_source_{m}.csv", index=False)
    fig_gap(summary)
    verdict = decide(summary)
    C.save_json(dict(config=CONFIG, verdict=verdict,
                     stage1_reference=dict(bagnet33_gap6_drop=0.054, resnet50_gap6_drop=0.293)),
                RES / "stage1b_report.json")

    pd.set_option("display.width", 220)
    print(summary[["model", "gap", "n_sources", "ctrl_acc", "cut_cost", "drop", "drop_lo", "drop_hi",
                   "excess_over_local", "excess_lo", "excess_hi", "decision"]].round(3).to_string(index=False))
    print("\n" + "=" * 78)
    print(f"  Stage 1 reference at 6 px: BagNet-33 drop 0.054, ResNet-50 drop 0.293")
    print(f"  STAGE 1b: {verdict['status']} | primary gap {verdict['primary_gap']} px | {verdict['primary_metric']}")
    print("=" * 78)

print("\nDone. Outputs in", RES, "and", FIGS)
