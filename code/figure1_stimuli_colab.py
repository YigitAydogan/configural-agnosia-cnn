"""
FIGURE 1: Stimulus examples for the paper (calibration set, frozen design)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (no GPU needed; about 2 minutes).
Uses only the CALIBRATION set rendered with the frozen design (frozen_calib: 6 px horizontal cut,
12 px grid gaps), so no hold-out image is shown. Reads agnosia_common.py from Drive (written by
every earlier stage), unpacks the calibration archive and draws one row per example object:

  Intact | Cut control (6 px) | Frankenstein | Grid control (12 px) | Scrambled

Examples: by default the first calibration TEST-half object (sorted by source id) in each of the
categories in CATEGORIES_SHOWN whose scramble changes the layout by at least 0.5 and whose grid
has 9 pieces. Set SOURCE_IDS to a list of source ids to choose the objects yourself.

Outputs: figures/paper/figure1_stimuli.pdf|png on Drive, and a copy in /content for download.
"""

import os
import sys
import shutil
import importlib
from pathlib import Path

if os.path.exists("/content") and not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

ROOT = Path(os.environ.get("AGNOSIA_ROOT", "/content/drive/MyDrive/AGNOSIA_CNN"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import agnosia_common as C
C = importlib.reload(C)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CATEGORIES_SHOWN = ["dog", "airplane", "bicycle"]
SOURCE_IDS = None            # e.g. ["coco_123456", "coco_234567", "coco_345678"]
MIN_LAYOUT_CHANGE = 0.5
STAGE = "frozen_calib"

COLUMNS = [("intact", "Intact"), ("cutH_intact", "Cut control"), ("frank_top", "Frankenstein"),
           ("cutG_intact", "Grid control"), ("scrambled", "Scrambled")]

# -----------------------------------------------------------------------------
# 1. Calibration stimuli
# -----------------------------------------------------------------------------
C.unpack_stimuli(STAGE)
man = pd.read_csv(C.STIM_DRIVE / f"{STAGE}_manifest.csv", low_memory=False)
tex = man[man.battery == "configural_tex"].copy()
assert not tex.empty, "No textured calibration images in the manifest."

if SOURCE_IDS is None:
    chosen = []
    for cat in CATEGORIES_SHOWN:
        sc = tex[(tex.category == cat) & (tex.condition == "scrambled") & (tex.split == "test")
                 & (tex.manip_strength >= MIN_LAYOUT_CHANGE) & (tex.n_pieces == 9)]
        ok = sorted(set(sc.source_id) & set(tex[(tex.condition == "frank_top") & (tex.manip_strength >= 0.1)].source_id))
        assert ok, f"No calibration test-half object of category {cat} meets the example criteria."
        chosen.append(ok[0])
else:
    chosen = list(SOURCE_IDS)
print("Examples:", ", ".join(chosen))


def pick(src, cond):
    r = tex[(tex.source_id == src) & (tex.condition == cond)]
    if cond == "scrambled":
        r = r.sort_values("manip_strength", ascending=False)
    else:
        r = r.sort_values("variant")
    assert len(r), f"{src} has no {cond} image."
    return np.array(Image.open(C.STIM_LOCAL / r.path.iloc[0]).convert("RGB"))


# -----------------------------------------------------------------------------
# 2. Figure
# -----------------------------------------------------------------------------
C.set_pub_style()
n_r, n_c = len(chosen), len(COLUMNS)
fig, axes = plt.subplots(n_r, n_c, figsize=(3.0 * n_c, 3.0 * n_r + 0.6))
axes = np.atleast_2d(axes)
for i, src in enumerate(chosen):
    cat = tex[tex.source_id == src].category.iloc[0]
    for j, (cond, lab) in enumerate(COLUMNS):
        ax = axes[i, j]
        ax.imshow(pick(src, cond))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.2)
            s.set_edgecolor("#8a8a8a")
        if i == 0:
            ax.text(0.5, 1.04, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=18)
        if j == 0:
            ax.set_ylabel(cat.capitalize(), fontsize=18)
fig.subplots_adjust(wspace=0.05, hspace=0.05)

out = C.FIG_DIR / "paper"
out.mkdir(parents=True, exist_ok=True)
C.save_figure(fig, out / "figure1_stimuli")
for ext in ("pdf", "png"):
    shutil.copy(out / f"figure1_stimuli.{ext}", f"/content/figure1_stimuli.{ext}")
print(f"Saved {out / 'figure1_stimuli.pdf'} and a copy in /content (download figure1_stimuli.pdf and send it).")
