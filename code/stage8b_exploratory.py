"""
STAGE 8b: Pre-listed exploratory analyses (hold-out set; no hypotheses)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 30 to 45 minutes on an A100). Lesion
blocks and model profiles are saved as they finish; rerunning skips finished parts.

The analysis plan lists these as exploratory ("labelled as such and carry no confirmatory claim"):
  A. Navon letters: global and local letter probes after lesions
  B. Frankenstein effects after lesions
  C. Per-category results (from the Stage 5 decisions; no model is run)
  D. The descriptive profile of every candidate model on the hold-out set (the seven unselected models,
     the two study models and the BagNet controls)
Same checks as Stages 5 and 6 before any hold-out image is used; every hold-out run is logged.

Lesion conditions for A and B (primary site, sets from lesion_sets_A1.json): intact, configural, local and
the first 20 damage-matched random sets, each under zero and mean ablation.

A. Navon. Linear probes (L2 logistic regression, 5-fold CV grouped by letter pair, as in Stage 6) on
   penultimate features, retrained per condition on the calibration Navon set: a global-letter probe
   (congruent, incongruent, global-only images) and a local-letter probe (congruent, incongruent,
   local-only images), both tested on hold-out incongruent letters (global and local letters differ).
   Global advantage = global minus local probe accuracy. Integrative agnosia is described as a loss of
   global form with local elements preserved, so the expected direction after a configural lesion is a
   fall in global advantage beyond random lesions. Bootstrap over hold-out letter pairs, stratified by
   global letter.
B. Frankenstein. Hold-out textured objects: 6 px horizontal-cut control versus the two Frankenstein
   variants (top or bottom half mirrored; variants with layout change below 10% excluded, as for scrambles).
   Frankenstein cost = control minus mean Frankenstein accuracy (network decisions, category-balanced).
   Expected direction after a configural lesion: a smaller cost (halves no longer combined) beyond random.
C. Per-category CR and the animate (bear, bird, cat, dog, elephant) versus inanimate split, intact and after
   the configural lesion, zero and mean ablation, from the Stage 5 checkpoints; floor rule per group.
D. Every model's intact CR, grid and scramble accuracy on the hold-out set, with the same bootstrap as Stage 5.
   The two study models' values must match Stage 5 (printed as a check).

Outputs: results/stage8b/*.csv|json|tex, figures/stage8b/*.pdf|png
"""

# =============================================================================
# 0. Setup and checks
# =============================================================================
import os
import sys
import json
import time
import hashlib
import importlib
from pathlib import Path

IN_COLAB = os.path.exists("/content") and "AGNOSIA_TEST" not in os.environ
if IN_COLAB and not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

AGNOSIA_COMMON_SRC = r'''"""
agnosia_common.py
Shared utilities for the CNN visual agnosia project (imported by every stage).

Place this file in DRIVE_ROOT (default /content/drive/MyDrive/AGNOSIA_CNN/).

Contents
  - paths, seeds, deterministic seed derivation
  - 16-category ImageNet mapping (Geirhos et al. model-vs-human convention)
  - dataset download / extraction with Drive caching
  - stimulus packing: stimuli are generated on local Colab disk and archived
    to Drive as one tar, because writing thousands of small PNGs to Drive is slow
  - model loading (BagNet-9/17/33, ResNet-50, extendable in Stage 2)
  - batched 16-class prediction
  - statistics: cluster bootstrap CIs, sign-flip tests, equivalence decisions
  - publication figure style
"""

import os
import io
import json
import random
import shutil
import tarfile
import zipfile
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

COMMON_VERSION = 5

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DRIVE_ROOT = Path(os.environ.get("AGNOSIA_ROOT", "/content/drive/MyDrive/AGNOSIA_CNN"))
LOCAL_ROOT = Path(os.environ.get("AGNOSIA_LOCAL", "/content/agnosia_local"))

RAW_DRIVE = DRIVE_ROOT / "raw"            # cached downloads (tar / zip)
STIM_DRIVE = DRIVE_ROOT / "stimuli"       # stimulus archives + manifests
RESULTS_DIR = DRIVE_ROOT / "results"
FIG_DIR = DRIVE_ROOT / "figures"

RAW_LOCAL = LOCAL_ROOT / "raw"            # extracted downloads
STIM_LOCAL = LOCAL_ROOT / "stimuli"       # extracted stimuli (paths in manifests are relative to this)

GLOBAL_SEED = 20261015
IMG_SIZE = 224

# ----------------------------------------------------------------------------
# 16-category mapping (Geirhos et al., model-vs-human, human_categories.py)
# ImageNet indices follow the standard sorted-WNID ordering.
# ----------------------------------------------------------------------------
CATEGORIES = sorted([
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
    "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck",
])
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
N_CAT = len(CATEGORIES)
CHANCE_16 = 1.0 / N_CAT

IMAGENET_16 = {
    "airplane": [404],
    "bear": [294, 295, 296, 297],
    "bicycle": [444, 671],
    "bird": [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 80, 81, 82, 83, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 135,
             136, 137, 138, 139, 140, 141, 142, 143, 144, 145],
    "boat": [472, 554, 625, 814, 914],
    "bottle": [440, 720, 737, 898, 899, 901, 907],
    "car": [436, 511, 817],
    "cat": [281, 282, 283, 284, 285, 286],
    "chair": [423, 559, 765, 857],
    "clock": [409, 530, 892],
    "dog": [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
            169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
            186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
            205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
            222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255, 256, 257, 259,
            261, 262, 263, 265, 266, 267, 268],
    "elephant": [385, 386],
    "keyboard": [508, 878],
    "knife": [499],
    "oven": [766],
    "truck": [555, 569, 656, 675, 717, 734, 864, 867],
}
assert sum(len(v) for v in IMAGENET_16.values()) == 207

MVH_URL = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{}.tar.gz"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------------
def ensure_dirs():
    for d in [RAW_DRIVE, STIM_DRIVE, RESULTS_DIR, FIG_DIR, RAW_LOCAL, STIM_LOCAL]:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def derive_seed(*keys):
    """Deterministic 32-bit seed from GLOBAL_SEED and arbitrary keys."""
    s = "|".join([str(GLOBAL_SEED)] + [str(k) for k in keys])
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
def download(url, dest, chunk=1 << 20):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done % (50 * chunk) < chunk:
                print(f"    {done / 1e6:0.1f} / {total / 1e6:0.1f} MB")
    tmp.rename(dest)
    return dest


def extract(archive, dest_dir):
    archive, dest_dir = Path(archive), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if str(archive).endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest_dir, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dest_dir)
    elif str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def get_mvh_dataset(name):
    """model-vs-human dataset (silhouette, edge, sketch, cue-conflict, ...). Returns local dir."""
    local = RAW_LOCAL / "model-vs-human" / name
    if local.exists() and any(local.rglob("*.png")):
        return local
    arc = download(MVH_URL.format(name), RAW_DRIVE / "model-vs-human" / f"{name}.tar.gz")
    extract(arc, RAW_LOCAL / "model-vs-human")
    assert local.exists(), f"Expected {local} after extraction"
    return local


def get_coco_annotations():
    """COCO 2017 instance annotations (no images needed). Returns local annotations dir."""
    local = RAW_LOCAL / "coco" / "annotations"
    if (local / "instances_val2017.json").exists():
        return local
    arc = download(COCO_ANN_URL, RAW_DRIVE / "coco" / "annotations_trainval2017.zip")
    extract(arc, RAW_LOCAL / "coco")
    return local


# ----------------------------------------------------------------------------
# Stimulus archives (local disk <-> Drive)
# ----------------------------------------------------------------------------
def stimuli_archive_path(stage):
    return STIM_DRIVE / f"{stage}_stimuli.tar"


def pack_stimuli(stage):
    src = STIM_LOCAL / stage
    arc = stimuli_archive_path(stage)
    arc.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp") / arc.name
    with tarfile.open(tmp, "w") as t:
        t.add(src, arcname=stage)
    shutil.copy(tmp, arc)
    tmp.unlink()
    return arc


def unpack_stimuli(stage):
    """Make STIM_LOCAL/<stage> available, extracting from Drive if needed."""
    dst = STIM_LOCAL / stage
    if dst.exists() and any(dst.rglob("*.png")):
        return dst
    arc = stimuli_archive_path(stage)
    assert arc.exists(), f"No stimulus archive at {arc}; run {stage} first."
    extract(arc, STIM_LOCAL)
    return dst


# ----------------------------------------------------------------------------
# Models and prediction
# ----------------------------------------------------------------------------
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_INFO = {
    # name: (display label, architecture family, role)
    "bagnet9": ("BagNet-9", "local", "control"),
    "bagnet17": ("BagNet-17", "local", "control"),
    "bagnet33": ("BagNet-33", "local", "control"),
    "resnet50": ("ResNet-50", "cnn", "candidate"),
    "resnet50_sin": ("ResNet-50 (SIN)", "cnn", "candidate"),
    "dino_resnet50": ("ResNet-50 (DINO)", "cnn", "candidate"),
    "resnet50_adv": ("ResNet-50 (adversarial)", "cnn", "candidate"),
    "resnet50_swsl": ("ResNet-50 (SWSL)", "cnn", "candidate"),
    "convnext_t": ("ConvNeXt-T", "cnn", "candidate"),
    "cornet_s": ("CORnet-S", "recurrent", "candidate"),
    "vit_b16": ("ViT-B/16", "transformer", "candidate"),
    "clip_vitb16": ("CLIP ViT-B/16", "transformer", "candidate"),
}
SIN_URL = ("https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/"
           "6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar")
DINO_BACKBONE_URL = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
DINO_LINEAR_URL = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
ROBUST_URL = "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps3.ckpt"
SWSL_TIMM = "resnet50.fb_swsl_ig1b_ft_in1k"


def _clip_zero_shot(device):
    """CLIP ViT-B/16 (OpenAI weights) with a zero-shot ImageNet head built from the 80 OpenAI
    prompt templates. Inputs arrive ImageNet-normalised; they are re-normalised for CLIP inside."""
    import torch
    import open_clip
    # OpenAI weights were trained with QuickGELU; the '-quickgelu' config matches them
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    clip = clip.eval().to(device)
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    W = open_clip.build_zero_shot_classifier(clip, tok, open_clip.IMAGENET_CLASSNAMES,
                                             open_clip.OPENAI_IMAGENET_TEMPLATES, num_classes_per_batch=20,
                                             device=device)

    class CLIPZeroShot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = clip.visual
            self.register_buffer("W", W.float())
            self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.register_buffer("clip_mean", torch.tensor(open_clip.OPENAI_DATASET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("clip_std", torch.tensor(open_clip.OPENAI_DATASET_STD).view(1, 3, 1, 1))
            self.scale = float(clip.logit_scale.exp().item())

        def forward(self, x):
            x = (x * self.in_std + self.in_mean - self.clip_mean) / self.clip_std
            f = self.visual(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            return self.scale * f @ self.W

    m = CLIPZeroShot()
    del clip.transformer, clip.token_embedding
    return m


def _strip_prefix(sd, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def load_model(name, device=None):
    """Pretrained ImageNet classifiers producing 1000-way logits.
    Extra packages: bagnets (BagNets), cornet (CORnet-S); install_model_packages() handles both."""
    import torch
    import torchvision
    from collections import OrderedDict
    device = device or get_device()
    if name in ("bagnet9", "bagnet17", "bagnet33"):
        from bagnets import pytorchnet
        model = getattr(pytorchnet, name)(pretrained=True)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == "resnet50_sin":
        model = torchvision.models.resnet50(weights=None)
        ck = torch.hub.load_state_dict_from_url(SIN_URL, map_location="cpu")
        model.load_state_dict(_strip_prefix(ck["state_dict"]))
    elif name == "dino_resnet50":
        backbone = torchvision.models.resnet50(weights=None)
        backbone.fc = torch.nn.Identity()
        sd = torch.hub.load_state_dict_from_url(DINO_BACKBONE_URL, map_location="cpu")
        missing, _ = backbone.load_state_dict(sd, strict=False)
        assert not missing, f"DINO backbone missing keys: {missing[:5]}"
        lin = _strip_prefix(torch.hub.load_state_dict_from_url(DINO_LINEAR_URL, map_location="cpu")["state_dict"])
        head = torch.nn.Linear(2048, 1000)
        head.load_state_dict({"weight": lin["linear.weight"], "bias": lin["linear.bias"]})
        model = torch.nn.Sequential(OrderedDict(backbone=backbone, head=head))
    elif name == "resnet50_adv":
        import os as _os
        dst = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50_l2_eps3.ckpt"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(ROBUST_URL, str(dst))
        ck = torch.load(dst, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        for pref in ("module.model.", "model."):
            sub = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
            if sub:
                break
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(sub)
    elif name == "resnet50_swsl":
        import timm
        model = timm.create_model(SWSL_TIMM, pretrained=True)
    elif name == "convnext_t":
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif name == "clip_vitb16":
        model = _clip_zero_shot(device)
    elif name == "cornet_s":
        import cornet
        model = cornet.cornet_s(pretrained=True, map_location="cpu").module
    elif name == "vit_b16":
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model {name}")
    return model.eval().to(device)


def install_model_packages():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/wielandbrendel/bag-of-local-features-models.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "git+https://github.com/dicarlolab/CORnet.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "open_clip_torch"], check=True)


def imagenet_preprocess():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImagePathDataset:
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def probs1000_to_16(p1000):
    """Mean softmax probability over the ImageNet classes of each category (model-vs-human)."""
    p1000 = np.asarray(p1000)
    return np.stack([p1000[:, IMAGENET_16[c]].mean(axis=1) for c in CATEGORIES], axis=1)


def predict_probs16(model, paths, device=None, batch_size=128, num_workers=2, model_name=None):
    """16-way probabilities. Any hold-out image in `paths` triggers the hold-out lock:
    model_name is then required, and models other than the local-only controls are refused
    until the analysis plan has been frozen (see freeze_plan)."""
    check_holdout_access(paths, model_name)
    import torch
    from torch.utils.data import DataLoader
    device = device or get_device()
    ds = ImagePathDataset(paths, imagenet_preprocess())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=(device.type == "cuda"))
    out = []
    with torch.no_grad():
        for x in dl:
            logits = model(x.to(device, non_blocking=True))
            p = torch.softmax(logits.float(), dim=1).cpu().numpy()
            out.append(probs1000_to_16(p))
    return np.concatenate(out, axis=0).astype(np.float64)


# ----------------------------------------------------------------------------
# Hold-out lock
#   The hold-out set is generated once (Stage 1c) and fingerprinted. Study models may
#   only be evaluated on it after the analysis plan is frozen with the same fingerprint.
#   Local-only control models (BagNets) are allowed, for the stimulus replication check.
#   Every hold-out evaluation is appended to an access log on Drive.
# ----------------------------------------------------------------------------
PLAN_DIR = DRIVE_ROOT / "plan"
HOLDOUT_STAGE = "holdout"
HOLDOUT_SPEC = PLAN_DIR / "holdout_spec.json"
FROZEN_PLAN = PLAN_DIR / "analysis_plan_frozen.json"
ACCESS_LOG = PLAN_DIR / "holdout_access_log.jsonl"
LOCAL_CONTROL_MODELS = ("bagnet9", "bagnet17", "bagnet33")


def is_holdout_path(p):
    return HOLDOUT_STAGE in Path(str(p)).parts


def fingerprint_files(root, rel_paths):
    """SHA-256 over sorted relative paths and the bytes of every file."""
    h = hashlib.sha256()
    for rel in sorted(str(r) for r in rel_paths):
        h.update(rel.encode())
        h.update(hashlib.sha256((Path(root) / rel).read_bytes()).digest())
    return h.hexdigest()


def holdout_status():
    if not HOLDOUT_SPEC.exists():
        return False, "no hold-out specification on Drive"
    if not FROZEN_PLAN.exists():
        return False, "analysis plan not frozen"
    spec, plan = load_json(HOLDOUT_SPEC), load_json(FROZEN_PLAN)
    if plan.get("holdout_fingerprint") != spec.get("fingerprint"):
        return False, "frozen plan does not match the hold-out fingerprint"
    return True, "unlocked"


def check_holdout_access(paths, model_name):
    n = sum(is_holdout_path(p) for p in paths)
    if n == 0:
        return
    if model_name is None:
        raise PermissionError("Hold-out images passed without model_name; refusing.")
    if model_name not in LOCAL_CONTROL_MODELS:
        ok, why = holdout_status()
        if not ok:
            raise PermissionError(f"Hold-out is locked for '{model_name}': {why}.")
    import time as _t
    import sys as _s
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps(dict(time=_t.strftime("%Y-%m-%d %H:%M:%S"), model=model_name, n_images=int(n),
                                script=str(_s.argv[0]))) + "\n")


def freeze_plan(plan):
    """Write the frozen analysis plan once. Refuses to overwrite."""
    import time as _t
    if FROZEN_PLAN.exists():
        raise FileExistsError(f"{FROZEN_PLAN} already exists; a frozen plan is never overwritten.")
    spec = load_json(HOLDOUT_SPEC)
    body = json.dumps(plan, sort_keys=True, default=str)
    save_json(dict(plan=plan, plan_sha256=hashlib.sha256(body.encode()).hexdigest(),
                   holdout_fingerprint=spec["fingerprint"], frozen_at=_t.strftime("%Y-%m-%d %H:%M:%S")),
              FROZEN_PLAN)
    return FROZEN_PLAN


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def bootstrap_mean_ci(x, n_boot=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI of the mean. x holds one value per independent unit
    (e.g. per source object, already averaged over variants), so this is a cluster bootstrap."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(GLOBAL_SEED)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def sign_flip_pvalue(d, n_perm=10000, rng=None):
    """Two-sided paired permutation test (random sign flips) on mean difference."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.allclose(d, 0):
        return 1.0
    rng = rng or np.random.default_rng(GLOBAL_SEED)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def equivalence_decision(lo, hi, mean, bound):
    """Interval-based decision with equivalence bound +-bound:
       'equivalent'      CI entirely inside (-bound, bound)
       'non-equivalent'  CI excludes zero and |mean| >= bound
       'inconclusive'    otherwise (CI too wide to decide)"""
    if not np.isfinite(lo):
        return "inconclusive"
    if lo > -bound and hi < bound:
        return "equivalent"
    if (lo > 0 or hi < 0) and abs(mean) >= bound:
        return "non-equivalent"
    return "inconclusive"


def cohens_dz(d):
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1) if d.size > 1 else np.nan
    return float(d.mean() / sd) if sd and sd > 0 else 0.0


# ----------------------------------------------------------------------------
# Publication figure style
# ----------------------------------------------------------------------------
SPINE_LW = 1.5
LEGEND_LW = 0.8   # thinner than spines

def set_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "axes.linewidth": SPINE_LW,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.formatter.use_mathtext": True,
    })


def style_axes(ax, xlim=None, xstep=None, ylim=None, ystep=None, decimals=None):
    """All four spines; numeric axes get labelled lower and upper bounds that are
    multiples of the tick step; decimals always shown with a leading zero."""
    from matplotlib.ticker import FormatStrFormatter
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(SPINE_LW)
    if ylim is not None and ystep is not None:
        lo = np.floor(ylim[0] / ystep + 1e-9) * ystep
        hi = np.ceil(ylim[1] / ystep - 1e-9) * ystep
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.round(np.arange(lo, hi + ystep / 2, ystep), 10))
        if decimals is not None:
            ax.yaxis.set_major_formatter(FormatStrFormatter(f"%0.{decimals}f"))
    if xlim is not None and xstep is not None:
        lo = np.floor(xlim[0] / xstep + 1e-9) * xstep
        hi = np.ceil(xlim[1] / xstep - 1e-9) * xstep
        ax.set_xlim(lo, hi)
        ax.set_xticks(np.round(np.arange(lo, hi + xstep / 2, xstep), 10))
    ax.tick_params(direction="out")


def outline_legend(leg):
    if leg is None:
        return
    fr = leg.get_frame()
    fr.set_edgecolor("black")
    fr.set_linewidth(LEGEND_LW)
    fr.set_alpha(1.0)


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
'''

COMMON_DIR = Path(os.environ.get("AGNOSIA_ROOT", "/content/drive/MyDrive/AGNOSIA_CNN"))
COMMON_DIR.mkdir(parents=True, exist_ok=True)
(COMMON_DIR / "agnosia_common.py").write_text(AGNOSIA_COMMON_SRC)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
import agnosia_common as C
C = importlib.reload(C)
if IN_COLAB:
    C.install_model_packages()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
sfx = "_smoke" if SMOKE else ""
SETS_SHA_REGISTERED = "9bb05eb1ee8729786246ff80f23575225c88f3271c4c6756a92525500c2fb904"
FROZEN = C.load_json(C.FROZEN_PLAN)
PLAN = FROZEN["plan"]
ok, why = C.holdout_status()
assert ok, f"Hold-out locked: {why}"
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{sfx}.json"
sets_sha = hashlib.sha256(SETS_PATH.read_bytes()).hexdigest()
if not SMOKE:
    assert sets_sha == SETS_SHA_REGISTERED, "lesion_sets_A1.json differs from the version published in v1.2."
SETS = C.load_json(SETS_PATH)

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    PROFILE_MODELS=["bagnet9", "bagnet17", "bagnet33", "resnet50", "resnet50_sin", "dino_resnet50", "resnet50_adv",
                    "resnet50_swsl", "convnext_t", "cornet_s", "vit_b16", "clip_vitb16"],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    ANIMATE=["bear", "bird", "cat", "dog", "elephant"],
    N_RANDOM=20,
    LAMBDAS=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], N_FOLDS=5, LBFGS_ITERS=150,
    N_BOOT=10000, BATCH=128, CHUNK=8,
)
if SMOKE:
    CONFIG.update(N_RANDOM=3, N_BOOT=300, BATCH=32, MIN_OBJECTS_PER_CATEGORY=2, LBFGS_ITERS=40, CHUNK=4,
                  PROFILE_MODELS=["bagnet33", "resnet50_swsl", "cornet_s"])

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage8b"
CKPT = RES / "checkpoints"
FIGS = C.FIG_DIR / "stage8b"
for d_ in [RES, CKPT, FIGS]:
    d_.mkdir(parents=True, exist_ok=True)
print(f"Stage 8b (exploratory) | lesion sets {sets_sha[:12]} | device {DEVICE}")

# =============================================================================
# 1. Images
# =============================================================================
if IN_COLAB:
    for st in ["stage1", C.HOLDOUT_STAGE]:
        C.unpack_stimuli(st)
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
s1 = pd.read_csv(C.STIM_DRIVE / "stage1_manifest.csv", low_memory=False)
spec = C.load_json(C.HOLDOUT_SPEC)
t0 = time.time()
assert C.fingerprint_files(C.STIM_LOCAL, hm.path.tolist()) == spec["fingerprint"], "Hold-out files changed since locking."
print(f"  hold-out fingerprint verified ({len(hm)} files, {time.time() - t0:0.0f} s)")

nav_c = s1[s1.battery == "navon"].reset_index(drop=True)
nav_h = hm[hm.battery == "navon"].reset_index(drop=True)
fr = hm[(hm.battery == "configural_tex") & hm.condition.isin(["cutH_intact", "frank_top", "frank_bottom"])].copy()
fr = fr[~(fr.condition.str.startswith("frank") & (fr.manip_strength < CONFIG["MIN_STRENGTH"]))]
fr = fr[~fr.category.isin(CONFIG["EXCLUDED"])]
keep = set(fr[fr.condition == "cutH_intact"].source_id) & set(fr[fr.condition != "cutH_intact"].source_id)
fr = fr[fr.source_id.isin(keep)]
npc = fr.drop_duplicates("source_id").category.value_counts()
fr = fr[~fr.category.isin(npc[npc < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
fr = fr.sort_values(["source_id", "condition"]).reset_index(drop=True)
assert not nav_c.path.map(C.is_holdout_path).any()

PARTS = [("nav_c", nav_c), ("nav_h", nav_h), ("fr", fr)]
ALL = pd.concat([d_.assign(part=k) for k, d_ in PARTS], ignore_index=True)
OFF, o = {}, 0
for k, d_ in PARTS:
    OFF[k] = (o, o + len(d_))
    o += len(d_)
size = 64 if SMOKE else C.IMG_SIZE
t0 = time.time()
IMGS = torch.from_numpy(np.stack([np.array(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)), dtype=np.uint8)
                                  for p in ALL.path])).permute(0, 3, 1, 2).contiguous()
HO_PATHS = [p for p in ALL.path if C.is_holdout_path(p)]
print(f"  Navon: calibration {len(nav_c)}, hold-out {len(nav_h)} | Frankenstein hold-out: {fr.source_id.nunique()} objects, "
      f"{len(fr)} images | loaded in {time.time() - t0:0.0f} s")

LETTERS = sorted(set(nav_c.global_letter.dropna()) | set(nav_c.local_letter.dropna()) - {""})
LETTERS = [l for l in LETTERS if isinstance(l, str) and l]
LID = {l: i for i, l in enumerate(LETTERS)}
lab = lambda s: np.array([LID.get(x, -1) if isinstance(x, str) else -1 for x in s])
NC_G, NC_L = lab(nav_c.global_letter), lab(nav_c.local_letter)
NH_G, NH_L = lab(nav_h.global_letter), lab(nav_h.local_letter)
NC_GROUP = nav_c.source_id.astype(str).to_numpy()
NH_INC = np.where(nav_h.condition == "navon_incongruent")[0]
NH_PAIR = nav_h.source_id.astype(str).to_numpy()[NH_INC]
NH_PAIR_G = nav_h.global_letter.to_numpy()[NH_INC]

FR_OBJ = sorted(fr.source_id.unique())
fo = {o_: i for i, o_ in enumerate(FR_OBJ)}
FR_O = fr.source_id.map(fo).to_numpy()
FR_Y = fr.category.map(C.CAT_TO_IDX).to_numpy()
FR_CAT = fr.drop_duplicates("source_id").set_index("source_id").loc[FR_OBJ, "category"].to_numpy()
FR_CTRL = np.array([np.where((fr.condition == "cutH_intact").to_numpy() & (FR_O == i))[0][0] for i in range(len(FR_OBJ))])
FR_AVG = np.zeros((len(fr), len(FR_OBJ)), np.float32)
for i in range(len(FR_OBJ)):
    idx = np.where(fr.condition.str.startswith("frank").to_numpy() & (FR_O == i))[0]
    FR_AVG[idx, i] = 1.0 / len(idx)

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def norm(x):
    return (x.to(DEVICE, non_blocking=True).float() / 255.0 - MEAN) / STD


# =============================================================================
# 2. Models and probes
# =============================================================================
def smoke_model(name):
    if name == "resnet50_swsl":
        import timm
        return timm.create_model("resnet50", pretrained=False).eval().to(DEVICE)
    if name == "cornet_s":
        import cornet
        return cornet.cornet_s(pretrained=False).module.eval().to(DEVICE)
    from bagnets import pytorchnet
    return pytorchnet.bagnet33(pretrained=False).eval().to(DEVICE)


class Split:
    def __init__(self, name):
        m = smoke_model(name) if SMOKE else C.load_model(name, DEVICE)
        if name == "resnet50_swsl":
            self.stages = [("stem", nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool)), ("layer1", m.layer1),
                           ("layer2", m.layer2), ("layer3", m.layer3), ("layer4", m.layer4)]
            self.feat = lambda x: m.forward_head(x, pre_logits=True)
            self.fc = m.fc
        else:
            self.stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.feat = lambda x: m.decoder.flatten(m.decoder.avgpool(x))
            self.fc = m.decoder.linear
        self.index = {n: i for i, (n, _) in enumerate(self.stages)}

    @torch.no_grad()
    def prefix(self, x, site):
        for i in range(self.index[site] + 1):
            x = self.stages[i][1](x)
        return x

    @torch.no_grad()
    def suffix(self, a, site):
        for i in range(self.index[site] + 1, len(self.stages)):
            a = self.stages[i][1](a)
        f = self.feat(a).float()
        return f, (torch.softmax(self.fc(f).float(), 1) @ M16).argmax(1)


N_NAV = OFF["nav_h"][1]          # Navon images come first; features are kept only for them


@torch.no_grad()
def run_conditions(net, site, conds, mean_act):
    n = len(ALL)
    feats = [None] * len(conds)
    decs = [np.zeros(n, np.int64) for _ in conds]
    mv = torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
    for b0 in range(0, n, CONFIG["BATCH"]):
        a = net.prefix(norm(IMGS[b0:b0 + CONFIG["BATCH"]]), site)
        for k, (ch, mode) in enumerate(conds):
            aa = a
            if len(ch):
                aa = a.clone()
                idx = torch.as_tensor(np.asarray(ch, dtype=np.int64), device=DEVICE)
                aa[:, idx] = 0 if mode == "zero" else mv[idx].view(1, -1, 1, 1).to(aa.dtype)
            f, p = net.suffix(aa, site)
            if feats[k] is None:
                feats[k] = torch.zeros(N_NAV, f.shape[1], device=DEVICE)
            m_ = min(len(f), max(0, N_NAV - b0))
            if m_ > 0:
                feats[k][b0:b0 + m_] = f[:m_]
            decs[k][b0:b0 + len(f)] = p.cpu().numpy()
    return feats, decs


def folds_for(groups):
    ug = np.unique(groups)
    rng = np.random.default_rng(C.derive_seed("stage8b_folds", len(ug)))
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % CONFIG["N_FOLDS"]))
    return np.array([fold_of[g] for g in groups])


def fit(X, Y, train_masks, lambdas, K, multilabel=False):
    """Fits one model per (mask, lambda) jointly. X (N, D) standardised; Y (N,) ints or (N, K) multi-hot.
    Returns W (M, D, K), b (M, K) for M = len(train_masks) * len(lambdas)."""
    N, D = X.shape
    masks = torch.as_tensor(np.repeat(train_masks, len(lambdas), axis=0), device=DEVICE, dtype=torch.float32)
    lam = torch.as_tensor(np.tile(lambdas, len(train_masks)), device=DEVICE, dtype=torch.float32)
    M = masks.shape[0]
    W = torch.zeros(M, D, K, device=DEVICE, requires_grad=True)
    b = torch.zeros(M, K, device=DEVICE, requires_grad=True)
    Yt = torch.as_tensor(Y, device=DEVICE)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=CONFIG["LBFGS_ITERS"], line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9, history_size=20)
    cnt = masks.sum(1)

    def closure():
        opt.zero_grad()
        z = torch.einsum("nd,mdk->mnk", X, W) + b[:, None]
        if multilabel:
            l = F.binary_cross_entropy_with_logits(z, Yt[None].expand(M, -1, -1).float(), reduction="none").mean(2)
        else:
            l = F.cross_entropy(z.permute(0, 2, 1), Yt[None].expand(M, -1), reduction="none")
        loss = ((l * masks).sum(1) / cnt + 0.5 * lam * (W ** 2).sum((1, 2))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    return [(x - mu) / sd for x in (Xtr,) + others]


def probe(Xtr, ytr, gtr, Xte, K, multilabel=False):
    """Grouped CV over lambdas, refit on all training data. Returns test scores (N_te, K) and lambda."""
    Xtr, Xte = standardise(Xtr, Xte)
    fold = folds_for(gtr)
    masks = np.stack([fold != f for f in range(CONFIG["N_FOLDS"])]).astype(np.float32)
    W, b = fit(Xtr, ytr, masks, CONFIG["LAMBDAS"], K, multilabel)
    L = len(CONFIG["LAMBDAS"])
    with torch.no_grad():
        z = torch.einsum("nd,mdk->mnk", Xtr, W) + b[:, None]
        score = np.zeros(L)
        for f in range(CONFIG["N_FOLDS"]):
            va = torch.as_tensor(fold == f, device=DEVICE)
            for li in range(L):
                zz = z[f * L + li][va]
                if multilabel:
                    yt = torch.as_tensor(ytr, device=DEVICE)[va]
                    score[li] += -F.binary_cross_entropy_with_logits(zz, yt).item()
                else:
                    score[li] += (zz.argmax(1).cpu().numpy() == ytr[fold == f]).mean()
    best = int(np.argmax(score[::-1]))
    best = L - 1 - best                                                  # ties go to stronger regularisation
    W1, b1 = fit(Xtr, ytr, np.ones((1, len(Xtr)), np.float32), [CONFIG["LAMBDAS"][best]], K, multilabel)
    with torch.no_grad():
        out = (Xte @ W1[0] + b1[0]).cpu().numpy()
    return out, CONFIG["LAMBDAS"][best]



def navon_outputs(feat):
    c0, c1 = OFF["nav_c"]
    h0, h1 = OFF["nav_h"]
    Fc, Fh = feat[c0:c1], feat[h0:h1]
    out = {}
    for which, ytr in [("global", NC_G), ("local", NC_L)]:
        ok_ = ytr >= 0
        sc, _ = probe(Fc[torch.as_tensor(ok_, device=DEVICE)], ytr[ok_], NC_GROUP[ok_],
                      Fh[torch.as_tensor(NH_INC, device=DEVICE)], len(LETTERS))
        out[f"{which}_pred"] = sc.argmax(1)
    return out


# =============================================================================
# 3. A and B: lesion runs (checkpointed per condition)
# =============================================================================
def conditions_for(name):
    e = SETS["models"][name]["sites"][SETS["models"][name]["primary_site"]]
    base = [("intact", [], "zero"), ("configural", e["configural"], "zero"), ("local", e["local"], "zero")]
    base += [(f"damage {i + 1}", s, "zero") for i, s in enumerate(e["random_damage_matched"][:CONFIG["N_RANDOM"]])]
    return base + [(f"{l} (mean)", s, "mean") for l, s, _ in base[1:]]


OUT = {}
for name in CONFIG["MODELS"]:
    print(f"\n[{C.MODEL_INFO[name][0]}] Navon and Frankenstein")
    site = SETS["models"][name]["primary_site"]
    mean_act = SETS["models"][name]["sites"][site]["mean_activation"]
    conds = conditions_for(name)
    OUT[name], todo = {}, []
    for lab_, ch, mode in conds:
        ck = CKPT / f"{name}_{lab_.replace(' ', '_').replace('(', '').replace(')', '')}{sfx}.npz"
        if ck.exists():
            z = np.load(ck)
            OUT[name][lab_] = {k: z[k] for k in z.files}
        else:
            todo.append((lab_, ch, mode, ck))
    if len(todo) < len(conds):
        print(f"  {len(conds) - len(todo)} of {len(conds)} conditions loaded from checkpoints")
    if not todo:
        continue
    C.check_holdout_access(HO_PATHS, name)
    net = Split(name)
    t0 = time.time()
    for c0 in range(0, len(todo), CONFIG["CHUNK"]):
        chunk = todo[c0:c0 + CONFIG["CHUNK"]]
        feats, decs = run_conditions(net, site, [(ch, mode) for _, ch, mode, _ in chunk], mean_act)
        for (lab_, _, _, ck), f, dd in zip(chunk, feats, decs):
            f0, f1 = OFF["fr"]
            o_ = dict(fr_dec=dd[f0:f1], **navon_outputs(f))
            np.savez_compressed(ck, **o_)
            OUT[name][lab_] = o_
        del feats
        torch.cuda.empty_cache()
        done = c0 + len(chunk)
        el = time.time() - t0
        print(f"    {done}/{len(todo)} conditions | {el / 60:0.1f} min | ETA {el / done * (len(todo) - done) / 60:0.1f} min")
    del net
    torch.cuda.empty_cache()


def strat_weights(strata, key):
    rng = np.random.default_rng(C.derive_seed("stage8b_boot", key))
    u = sorted(set(strata))
    W = np.zeros((CONFIG["N_BOOT"], len(strata)), np.float32)
    w0 = np.zeros(len(strata), np.float32)
    for c in u:
        g = np.where(strata == c)[0]
        s = 1.0 / (len(g) * len(u))
        w0[g] = s
        dr = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
        cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
        np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), dr.ravel()), 1)
        W[:, g] = cnt * s
    return w0, W


# Navon: units are hold-out incongruent images, resampled by letter pair within global letter
pairs = sorted(set(NH_PAIR))
pair_idx = {p: i for i, p in enumerate(pairs)}
PAIR_OF = np.array([pair_idx[p] for p in NH_PAIR])
PAIR_G = np.array([NH_PAIR_G[np.where(NH_PAIR == p)[0][0]] for p in pairs])
WN = strat_weights(PAIR_G, "navon")
AVG_PAIR = np.zeros((len(NH_INC), len(pairs)), np.float32)
for j, p in enumerate(PAIR_OF):
    AVG_PAIR[j, p] = 1.0
AVG_PAIR /= AVG_PAIR.sum(0, keepdims=True)
WF = strat_weights(FR_CAT, "frankenstein")


def navon_measures(o_):
    g = ((o_["global_pred"] == NH_G[NH_INC]).astype(np.float32)) @ AVG_PAIR
    l_ = ((o_["local_pred"] == NH_L[NH_INC]).astype(np.float32)) @ AVG_PAIR
    w0, W = WN
    return dict(global_acc=(w0 @ g, W @ g), local_acc=(w0 @ l_, W @ l_), global_advantage=(w0 @ (g - l_), W @ (g - l_)))


def frank_measures(o_):
    corr = (o_["fr_dec"] == FR_Y).astype(np.float32)
    ctrl, frk = corr[FR_CTRL], corr @ FR_AVG
    w0, W = WF
    return dict(control_acc=(w0 @ ctrl, W @ ctrl), frank_acc=(w0 @ frk, W @ frk), frank_cost=(w0 @ (ctrl - frk), W @ (ctrl - frk)))


ab_rows, ab_con = [], []
for name in CONFIG["MODELS"]:
    for mode, tag in [("zero", ""), ("mean", " (mean)")]:
        base = OUT[name]["intact"]
        conf = OUT[name][f"configural{tag}"]
        loc = OUT[name][f"local{tag}"]
        rnd = [OUT[name][f"damage {i + 1}{tag}"] for i in range(CONFIG["N_RANDOM"]) if f"damage {i + 1}{tag}" in OUT[name]]
        for fam, fn in [("Navon", navon_measures), ("Frankenstein", frank_measures)]:
            mb, mc, ml = fn(base), fn(conf), fn(loc)
            mr = [fn(r) for r in rnd]
            for meas in mb:
                rp = np.array([x[meas][0] for x in mr])
                ab_rows.append(dict(model=name, ablation=mode, analysis=fam, measure=meas, intact=float(mb[meas][0]),
                                    configural=float(mc[meas][0]), local=float(ml[meas][0]), random_mean=float(rp.mean()),
                                    random_lo=float(np.quantile(rp, 0.025)), random_hi=float(np.quantile(rp, 0.975))))
            key = "global_advantage" if fam == "Navon" else "frank_cost"
            dc_p, dc_b = mc[key][0] - mb[key][0], mc[key][1] - mb[key][1]
            rb = np.stack([x[key][1] for x in mr], 1)
            rp = np.array([x[key][0] for x in mr])
            diff_b = dc_b - np.mean(rb - mb[key][1][:, None], 1)
            ab_con.append(dict(model=name, ablation=mode, analysis=fam,
                               contrast=f"change in {key.replace('_', ' ')}: configural minus mean random",
                               estimate=float(dc_p - np.mean(rp - mb[key][0])), lo=float(np.quantile(diff_b, 0.025)),
                               hi=float(np.quantile(diff_b, 0.975))))
            dl_b = (ml[key][1] - mb[key][1])
            ab_con.append(dict(model=name, ablation=mode, analysis=fam,
                               contrast=f"change in {key.replace('_', ' ')}: configural minus local",
                               estimate=float(dc_p - (ml[key][0] - mb[key][0])),
                               lo=float(np.quantile(dc_b - dl_b, 0.025)), hi=float(np.quantile(dc_b - dl_b, 0.975))))
ab, abc = pd.DataFrame(ab_rows), pd.DataFrame(ab_con)

# =============================================================================
# 4. C: per-category results from the Stage 5 checkpoints (no model is run)
# =============================================================================
d5 = hm[(hm.battery == "configural_tex") & hm.condition.isin(["cutG_intact", "scrambled"])].copy()
d5 = d5[~((d5.condition == "scrambled") & (d5.manip_strength < CONFIG["MIN_STRENGTH"]))]
d5 = d5[~d5.category.isin(CONFIG["EXCLUDED"])]
k5 = set(d5[d5.condition == "cutG_intact"].source_id) & set(d5[d5.condition == "scrambled"].source_id)
d5 = d5[d5.source_id.isin(k5)]
n5 = d5.drop_duplicates("source_id").category.value_counts()
d5 = d5[~d5.category.isin(n5[n5 < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
d5 = d5.sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
O5 = sorted(d5.source_id.unique())
o5 = {x: i for i, x in enumerate(O5)}
OBJ5 = d5.source_id.map(o5).to_numpy()
CAT5 = d5.drop_duplicates("source_id").set_index("source_id").loc[O5, "category"].to_numpy()
G5 = np.array([np.where((d5.condition == "cutG_intact").to_numpy() & (OBJ5 == i))[0][0] for i in range(len(O5))])
S5 = np.zeros((len(d5), len(O5)), np.float32)
for i in range(len(O5)):
    idx = np.where((d5.condition == "scrambled").to_numpy() & (OBJ5 == i))[0]
    S5[idx, i] = 1.0 / len(idx)
CK5 = C.RESULTS_DIR / "stage5" / "checkpoints"
cat_rows = []
groups = {c: [c] for c in sorted(set(CAT5))}
groups.update(animate=[c for c in CONFIG["ANIMATE"] if c in set(CAT5)],
              inanimate=[c for c in sorted(set(CAT5)) if c not in CONFIG["ANIMATE"]])
for name in CONFIG["MODELS"]:
    for block, mode in [("primary_zero", "zero"), ("primary_mean", "mean")]:
        f = CK5 / f"{name}_{block}{sfx}.npz"
        if not f.exists():
            print(f"  Stage 5 checkpoint missing: {f.name}; per-category results skipped for it")
            continue
        z = np.load(f, allow_pickle=True)
        labels, corr = list(z["labels"]), z["correct"]
        assert corr.shape[1] == len(d5), "Stage 5 checkpoint does not match the hold-out image list"
        for gname, cats in groups.items():
            members = np.where(np.isin(CAT5, cats))[0]
            w0, W = strat_weights(CAT5[members], f"cat_{gname}")
            res = {}
            for lab_ in ["intact", "configural", "local"]:
                c_ = corr[labels.index(lab_)]
                g = c_[G5][members].astype(np.float32)
                s_ = (c_.astype(np.float32) @ S5)[members]
                AG, AS = W @ g, W @ s_
                ag, asc = w0 @ g, w0 @ s_
                floor = np.quantile(AG, 0.025) < 2 * C.CHANCE_16
                with np.errstate(divide="ignore", invalid="ignore"):
                    crb = (AG - AS) / (AG - C.CHANCE_16)
                res[lab_] = (np.nan if floor else (ag - asc) / (ag - C.CHANCE_16), crb, floor, ag, asc)
            for lab_, (v, crb, floor, ag, asc) in res.items():
                row = dict(model=name, ablation=mode, group=gname, n_objects=len(members), condition=lab_, A_grid=float(ag),
                           A_scr=float(asc), CR=v, floor=bool(floor),
                           CR_lo=np.nan if floor else float(np.nanquantile(crb, 0.025)),
                           CR_hi=np.nan if floor else float(np.nanquantile(crb, 0.975)))
                if lab_ != "intact" and not floor and not res["intact"][2]:
                    dlt = crb - res["intact"][1]
                    row.update(dCR=float(v - res["intact"][0]), dCR_lo=float(np.nanquantile(dlt, 0.025)),
                               dCR_hi=float(np.nanquantile(dlt, 0.975)))
                cat_rows.append(row)
catdf = pd.DataFrame(cat_rows)

# =============================================================================
# 5. D: descriptive profile of every candidate model (intact, hold-out configural battery)
# =============================================================================
wP = strat_weights(CAT5, "profile")
prof = []
for name in CONFIG["PROFILE_MODELS"]:
    ck = CKPT / f"profile_{name}{sfx}.npz"
    if ck.exists():
        corr = np.load(ck)["correct"]
    else:
        if SMOKE:
            net = smoke_model(name if name in ("resnet50_swsl", "cornet_s") else "bagnet33")
        else:
            net = C.load_model(name, DEVICE)
        paths = [str(C.STIM_LOCAL / p) for p in d5.path]
        P = C.predict_probs16(net, paths, DEVICE, batch_size=CONFIG["BATCH"], num_workers=2, model_name=name)
        corr = P.argmax(1) == d5.category.map(C.CAT_TO_IDX).to_numpy()
        np.savez_compressed(ck, correct=corr)
        del net
        torch.cuda.empty_cache()
    g = corr[G5].astype(np.float32)
    s_ = corr.astype(np.float32) @ S5
    w0, W = wP
    AG, AS = W @ g, W @ s_
    with np.errstate(divide="ignore", invalid="ignore"):
        crb = (AG - AS) / (AG - C.CHANCE_16)
    ag, asc = w0 @ g, w0 @ s_
    floor = np.quantile(AG, 0.025) < 2 * C.CHANCE_16
    prof.append(dict(model=name, label=C.MODEL_INFO[name][0], family=C.MODEL_INFO[name][1],
                     role=("study (primary)" if name == CONFIG["MODELS"][0] else "study (secondary)" if name == CONFIG["MODELS"][1]
                           else C.MODEL_INFO[name][2]),
                     A_grid=float(ag), A_scr=float(asc), loss=float(ag - asc),
                     CR=np.nan if floor else float((ag - asc) / (ag - C.CHANCE_16)),
                     CR_lo=np.nan if floor else float(np.nanquantile(crb, 0.025)),
                     CR_hi=np.nan if floor else float(np.nanquantile(crb, 0.975))))
    print(f"  profile {C.MODEL_INFO[name][0]:<24} grid {ag:0.3f} | scrambled {asc:0.3f} | CR "
          f"{'floor' if floor else f'{(ag - asc) / (ag - C.CHANCE_16):0.3f}'}")
profdf = pd.DataFrame(prof)
s5sum = C.RESULTS_DIR / "stage5" / f"conditions{sfx}.csv"
if s5sum.exists():
    c5 = pd.read_csv(s5sum)
    for name in CONFIG["MODELS"]:
        r5 = c5[(c5.model == name) & (c5.block == "primary_zero") & (c5.condition == "intact")]
        rp = profdf[profdf.model == name]
        if len(r5) and len(rp):
            print(f"  check {C.MODEL_INFO[name][0]}: CR here {rp.CR.iloc[0]:0.4f} vs Stage 5 {r5.CR.iloc[0]:0.4f}")

# =============================================================================
# 6. Save, table, figure
# =============================================================================
ab.to_csv(RES / f"navon_frankenstein_measures{sfx}.csv", index=False)
abc.to_csv(RES / f"navon_frankenstein_contrasts{sfx}.csv", index=False)
catdf.to_csv(RES / f"per_category{sfx}.csv", index=False)
profdf.to_csv(RES / f"model_profile{sfx}.csv", index=False)
C.save_json(dict(config=CONFIG, lesion_sets_sha256=sets_sha, letters=LETTERS, run_at=time.strftime("%Y-%m-%d %H:%M:%S")),
            RES / f"stage8b_summary{sfx}.json")
fmtci = lambda m, lo, hi: "floor" if not np.isfinite(m) else f"{m:0.3f} [{lo:0.3f}, {hi:0.3f}]"
lines = [r"\begin{tabular}{lllrrr}", r"\hline", r"Model & Family & Role & Grid acc. & Scrambled acc. & CR [95\% CI] \\", r"\hline"]
for r in profdf.itertuples():
    lines.append(f"{r.label} & {r.family} & {r.role} & {r.A_grid:0.3f} & {r.A_scr:0.3f} & {fmtci(r.CR, r.CR_lo, r.CR_hi)} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage8b_model_profile.tex").write_text("\n".join(lines))


def tidy(ax, dec):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{dec}f}"))


def figure():
    """Per-category CR, intact and after the configural lesion (mean ablation, which avoids the readout floor)."""
    C.set_pub_style()
    fig, axes = plt.subplots(len(CONFIG["MODELS"]), 1, figsize=(17, 6.2 * len(CONFIG["MODELS"])))
    axes = np.atleast_1d(axes)
    cats = sorted(set(CAT5))
    order = [c for c in cats if c in CONFIG["ANIMATE"]] + [c for c in cats if c not in CONFIG["ANIMATE"]]
    sub = catdf[(catdf.ablation == "mean") & catdf.group.isin(order)]
    vals = pd.concat([sub.CR_lo, sub.CR_hi, sub.CR]).dropna().to_numpy()
    lo = min(-0.2, np.floor(vals.min() / 0.2) * 0.2) if len(vals) else -0.2
    hi = max(1.0, np.ceil(vals.max() / 0.2) * 0.2) if len(vals) else 1.0
    for p, (ax, name) in enumerate(zip(axes, CONFIG["MODELS"])):
        g = sub[sub.model == name]
        for j, (cond, col, mk, dx) in enumerate([("intact", "#000000", "o", -0.15), ("configural", "#2a78d6", "s", 0.15)]):
            r = g[g.condition == cond].set_index("group").reindex(order)
            x = np.arange(len(order)) + dx
            okm = np.isfinite(r.CR.to_numpy())
            ax.errorbar(x[okm], r.CR.to_numpy()[okm], yerr=[(r.CR - r.CR_lo).to_numpy()[okm], (r.CR_hi - r.CR).to_numpy()[okm]],
                        fmt=mk, ms=8, capsize=4, lw=1.6, color=col, mfc="white" if cond == "intact" else col,
                        label=("Intact" if cond == "intact" else "Configural lesion (mean ablation)") if p == 0 else None)
        ax.axvline(len([c for c in order if c in CONFIG["ANIMATE"]]) - 0.5, color="#8a8a8a", lw=1, ls="--")
        ax.axhline(0, color="#000000", lw=0.8)
        C.style_axes(ax, ylim=(lo, hi), ystep=0.2, decimals=1)
        tidy(ax, 1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel(f"Configural reliance\n{C.MODEL_INFO[name][0]}")
        ax.yaxis.set_label_coords(-0.07, 0.5)
        ax.text(-0.07, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    C.outline_legend(leg)
    fig.subplots_adjust(hspace=0.55, bottom=0.14)
    C.save_figure(fig, FIGS / f"stage8b_categories{sfx}")
    plt.close(fig)


figure()

pd.set_option("display.width", 240)
show = lambda df: df.round(4).astype(object).where(df.notna(), "").to_string(index=False)
print("\nA and B. Navon probes and Frankenstein cost (intact, configural, local, random = mean and 95% range of sets)")
print(show(ab))
print("\nContrasts (configural lesion, change from intact)")
print(show(abc))
print("\nC. Animate versus inanimate (Stage 5 decisions)")
print(show(catdf[catdf.group.isin(["animate", "inanimate"])]))
print("\nD. Model profile on the hold-out set (intact)")
print(show(profdf))
print("\nDone. Outputs in", RES, "and", FIGS)
