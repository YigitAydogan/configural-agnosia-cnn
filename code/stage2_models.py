"""
STAGE 2: Model zoo baseline and model selection
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU runtime, about 25 minutes).
Outputs are tables only: the selection table (LaTeX) carries the key numbers, and per-condition
accuracies are saved as CSV for the final figures.
The shared module (agnosia_common.py) is embedded and written to Drive on each run.

Data: the LOCALIZER half of the frozen calibration set only (textured and silhouette
renderings), the localizer half of the Stage 1 overlapping-figure pairs, and the public
model-vs-human reference sets (cue-conflict, silhouette, edge, sketch). The test half of the
calibration set and the hold-out set are not touched.

Candidates (nine), chosen to answer the standard reviewer questions in two series
  Training series, architecture fixed (ResNet-50):
    supervised ImageNet; Stylized-ImageNet (shape-biased); DINO (self-supervised, published
    linear read-out); adversarially trained (L2, eps 3); SWSL (semi-weakly supervised on
    940M Instagram images, then ImageNet), i.e. objective, robustness and data scale
  Architecture series:
    ConvNeXt-T (modern convolutional design), CORnet-S (recurrent, brain-inspired),
    ViT-B/16 (transformer, ImageNet), CLIP ViT-B/16 (vision-language, 400M image-text pairs,
    zero-shot read-out from the 80 OpenAI ImageNet prompts)
  BagNet-33 is run as the local-only baseline.

Measures (per model)
  - accuracy per condition (intact, cut controls, Frankenstein, scrambled), both renderings
  - scramble effect in excess of BagNet-33 (textured, 12 px gaps, paired per source)
  - Frankenstein effect: in excess of BagNet-33 (textured) and raw (silhouette, where local
    equivalence holds by construction)
  - shape bias on cue-conflict images (fraction of shape decisions among shape or texture decisions)
  - accuracy on silhouette, edge and sketch reference sets
  - overlapping figures: rate at which the top two categories are exactly the two objects,
    side by side minus overlapping (scale-matched)

Selection rule (fixed before running)
  A candidate qualifies if
    (1) headroom: the lower 95% CI of accuracy on both textured cut controls (horizontal cut
        and 12 px grid) exceeds 2 x chance, so lesions have room to act, and
    (2) configural sensitivity: the lower 95% CI of its scramble excess over BagNet-33 exceeds
        0.05, the bound below which an effect was treated as negligible for BagNet-33.
  Primary model: the qualifier with the highest lower CI of scramble excess.
  Secondary model: the best qualifier from a different architecture family (cnn, recurrent,
  transformer); if none qualifies, the next best qualifier overall.
  If no candidate qualifies, the project pivots to a model-comparison paper (planned gate).
  Frankenstein, shape bias, reference sets and overlap are descriptive, not selection criteria.
"""

# =============================================================================
# 0. Setup
# =============================================================================
import os
import sys
import json
import time
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

RUN_MODELS = os.environ.get("AGNOSIA_NO_MODELS") is None
CONFIG = dict(
    CANDIDATES=["resnet50", "resnet50_sin", "dino_resnet50", "resnet50_adv", "resnet50_swsl",
                "convnext_t", "cornet_s", "vit_b16", "clip_vitb16"],
    BASELINE="bagnet33",
    SPLIT="localizer",
    FLOOR_MULT=2.0,
    MIN_EXCESS=0.05,
    N_BOOT=10000,
    BATCH=128,
)
FORCE_PREDICTIONS = False
CACHE_TAG = {"clip_vitb16": "_quickgelu"}   # loader fixed in shared module v5; forces a fresh CLIP run
HUMAN_SHAPE_BIAS = 0.96   # Geirhos et al. (2019), cue-conflict, human observers

C.ensure_dirs()
C.set_seed()
RES = C.RESULTS_DIR / "stage2"
FIGS = C.FIG_DIR / "stage2"
for d in [RES, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
FLOOR = CONFIG["FLOOR_MULT"] * C.CHANCE_16
LABEL = {k: v[0] for k, v in C.MODEL_INFO.items()}
FAMILY = {k: v[1] for k, v in C.MODEL_INFO.items()}
print(f"Stage 2 | shared module v{C.COMMON_VERSION} | {len(CONFIG['CANDIDATES'])} candidates")


# =============================================================================
# 1. Evaluation set (localizer split only)
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("frozen_calib")
    C.unpack_stimuli("stage1")
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
s1 = pd.read_csv(C.STIM_DRIVE / "stage1_manifest.csv", low_memory=False)

cfg = fc[fc.split == CONFIG["SPLIT"]].copy()
cfg["group"] = cfg.battery
ov = s1[(s1.battery == "overlap") & s1.condition.isin(["overlap_large", "overlap_small", "side_by_side"])
        & (s1.split == CONFIG["SPLIT"])].copy()
ov["group"] = "overlap"
ref = s1[s1.battery == "reference"].copy()
ref["group"] = "reference"
cols = ["path", "group", "battery", "condition", "variant", "source_set", "source_id", "category", "split",
        "manip_strength", "category_b", "texture_category"]
ev = pd.concat([d.reindex(columns=cols) for d in (cfg, ov, ref)], ignore_index=True)
assert not ev.path.map(C.is_holdout_path).any()
assert not (ev.group.isin(["configural", "configural_tex"]) & (ev.split != CONFIG["SPLIT"])).any()
print(ev.groupby(["group", "condition"]).size().to_string())


# =============================================================================
# 2. Predictions
# =============================================================================
def run_predictions():
    import torch
    device = C.get_device()
    paths = [str(C.STIM_LOCAL / p) for p in ev.path]
    preds = {}
    for name in [CONFIG["BASELINE"]] + CONFIG["CANDIDATES"]:
        out = RES / f"predictions_{name}{CACHE_TAG.get(name, '')}.csv"
        if out.exists() and not FORCE_PREDICTIONS:
            df = pd.read_csv(out, low_memory=False)
            if df.path.tolist() == ev.path.tolist():
                preds[name] = df
                print(f"  {name}: cached")
                continue
        t0 = time.time()
        model = C.load_model(name, device)
        p16 = C.predict_probs16(model, paths, device, batch_size=CONFIG["BATCH"], model_name=name)
        del model
        torch.cuda.empty_cache()
        df = ev.copy()
        y = df.category.map(C.CAT_TO_IDX).to_numpy()
        df["pred"] = [C.CATEGORIES[i] for i in p16.argmax(1)]
        df["correct"] = (p16.argmax(1) == y).astype(int)
        for i, c in enumerate(C.CATEGORIES):
            df[f"p_{c}"] = p16[:, i]
        df.to_csv(out, index=False)
        preds[name] = df
        print(f"  {name}: {time.time() - t0:0.0f} s")
    return preds


# =============================================================================
# 3. Measures
# =============================================================================
COND_ORDER = ["intact", "cutH_intact", "frankenstein", "cutG_intact", "scrambled"]
COND_LABEL = {"intact": "Intact", "cutH_intact": "Cut (H)", "frankenstein": "Frankenstein",
              "cutG_intact": "Cut (grid)", "scrambled": "Scrambled"}
MANIP = ["frank_top", "frank_bottom", "scrambled"]


def wide_acc(df, battery):
    d = df[df.group == battery].copy()
    d = d[~(d.condition.isin(MANIP) & (d.manip_strength < 0.10))]
    d["cond"] = d.condition.replace({"frank_top": "frankenstein", "frank_bottom": "frankenstein"})
    g = d.groupby(["source_id", "category", "cond"]).correct.mean().reset_index()
    w = g.pivot(index="source_id", columns="cond", values="correct")
    cat = g.drop_duplicates("source_id").set_index("source_id").category
    return w, cat


def drop(w, ctrl, manip):
    both = w[[ctrl, manip]].dropna().index
    return (w.loc[both, ctrl] - w.loc[both, manip])


def cat_balanced(series, cat):
    return float(series.groupby(cat.reindex(series.index)).mean().mean())


def relative_loss(w, base_w, rng, B):
    """Post hoc (added after the first Stage 2 run): scramble drop as a proportion of the
    above-chance accuracy on the grid control, (ctrl - scr) / (ctrl - chance), and its excess over
    BagNet-33. Absolute drops are capped by control accuracy, so models with little headroom cannot
    show large absolute drops; this measure removes that cap. Sources are resampled jointly."""
    cols = ["cutG_intact", "scrambled"]
    idx = w[cols].dropna().index.intersection(base_w[cols].dropna().index)
    c, s = w.loc[idx, cols[0]].to_numpy(), w.loc[idx, cols[1]].to_numpy()
    bc, bs = base_w.loc[idx, cols[0]].to_numpy(), base_w.loc[idx, cols[1]].to_numpy()
    ch = C.CHANCE_16

    def stat(I):
        cm, sm, bcm, bsm = c[I].mean(-1), s[I].mean(-1), bc[I].mean(-1), bs[I].mean(-1)
        rel = (cm - sm) / np.maximum(cm - ch, 1e-9)
        brel = (bcm - bsm) / np.maximum(bcm - ch, 1e-9)
        return rel, rel - brel

    rel, ex = stat(np.arange(len(idx)))
    I = rng.integers(0, len(idx), size=(B, len(idx)))
    br, be = stat(I)
    q = lambda x: np.quantile(x, [0.025, 0.975])
    return float(rel), *q(br), float(ex), *q(be)


def measures(preds):
    rng = np.random.default_rng(C.derive_seed("stage2"))
    B = CONFIG["N_BOOT"]
    base_tex, _ = wide_acc(preds[CONFIG["BASELINE"]], "configural_tex")
    base_scr = drop(base_tex, "cutG_intact", "scrambled")
    base_frk = drop(base_tex, "cutH_intact", "frankenstein")
    rows, acc_rows = [], []
    for name, df in preds.items():
        r = dict(model=name, label=LABEL[name], family=FAMILY[name])
        for battery in ["configural", "configural_tex"]:
            w, cat = wide_acc(df, battery)
            for cond in COND_ORDER:
                m, lo, hi = C.bootstrap_mean_ci(w[cond], B, rng=rng)
                acc_rows.append(dict(model=name, battery=battery, condition=cond, acc=m, lo=lo, hi=hi,
                                     n=int(w[cond].notna().sum())))
            if battery == "configural_tex":
                tex_w, tex_cat = w, cat
            else:
                sil = drop(w, "cutH_intact", "frankenstein")
                r["sil_frank_drop"], r["sil_frank_lo"], r["sil_frank_hi"] = C.bootstrap_mean_ci(sil, B, rng=rng)
        for key, ctrl in [("cutH", "cutH_intact"), ("cutG", "cutG_intact"), ("intact", "intact")]:
            r[f"tex_{key}_acc"], r[f"tex_{key}_lo"], r[f"tex_{key}_hi"] = C.bootstrap_mean_ci(tex_w[ctrl], B, rng=rng)
        scr = drop(tex_w, "cutG_intact", "scrambled")
        frk = drop(tex_w, "cutH_intact", "frankenstein")
        r["scr_drop"], r["scr_drop_lo"], r["scr_drop_hi"] = C.bootstrap_mean_ci(scr, B, rng=rng)
        r["frk_drop"], r["frk_drop_lo"], r["frk_drop_hi"] = C.bootstrap_mean_ci(frk, B, rng=rng)
        ex_s = (scr - base_scr.reindex(scr.index)).dropna()
        ex_f = (frk - base_frk.reindex(frk.index)).dropna()
        r["scr_excess"], r["scr_excess_lo"], r["scr_excess_hi"] = C.bootstrap_mean_ci(ex_s, B, rng=rng)
        r["frk_excess"], r["frk_excess_lo"], r["frk_excess_hi"] = C.bootstrap_mean_ci(ex_f, B, rng=rng)
        r["scr_excess_cat_balanced"] = cat_balanced(ex_s, tex_cat)
        (r["scr_rel"], r["scr_rel_lo"], r["scr_rel_hi"],
         r["scr_rel_excess"], r["scr_rel_excess_lo"], r["scr_rel_excess_hi"]) = relative_loss(tex_w, base_tex, rng, B)
        r["n_scr"], r["n_frk"] = len(ex_s), len(ex_f)

        # shape bias (cue-conflict, shape and texture categories differ)
        cc = df[(df.condition == "ref_cue-conflict") & (df.category != df.texture_category)]
        shape_hit = (cc.pred == cc.category).to_numpy()
        tex_hit = (cc.pred == cc.texture_category).to_numpy()
        keep = shape_hit | tex_hit
        sb = shape_hit[keep].astype(float)
        r["shape_bias"], r["shape_bias_lo"], r["shape_bias_hi"] = C.bootstrap_mean_ci(sb, B, rng=rng)
        r["cue_conflict_decided_frac"] = float(keep.mean())
        for refname in ["silhouette", "edge", "sketch"]:
            x = df[df.condition == f"ref_{refname}"].correct
            r[f"ref_{refname}_acc"], r[f"ref_{refname}_lo"], r[f"ref_{refname}_hi"] = C.bootstrap_mean_ci(x, B, rng=rng)

        # overlapping figures: both objects are the top two of 16
        o = df[df.group == "overlap"].copy()
        P = o[[f"p_{c}" for c in C.CATEGORIES]].to_numpy()
        top2 = np.argsort(-P, axis=1)[:, :2]
        a = o.category.map(C.CAT_TO_IDX).to_numpy()
        b = o.category_b.map(C.CAT_TO_IDX).to_numpy()
        o["both"] = (((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a))).astype(float)
        ow = o.pivot_table(index="source_id", columns="condition", values="both")
        for cond in ["side_by_side", "overlap_small", "overlap_large"]:
            r[f"ov_{cond}"] = float(ow[cond].mean())
        r["ov_cost"], r["ov_cost_lo"], r["ov_cost_hi"] = C.bootstrap_mean_ci(ow.side_by_side - ow.overlap_small, B, rng=rng)
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(acc_rows)


# =============================================================================
# 4. Selection
# =============================================================================
def select(summary):
    s = summary[summary.model.isin(CONFIG["CANDIDATES"])].copy()
    s["headroom_ok"] = (s.tex_cutH_lo > FLOOR) & (s.tex_cutG_lo > FLOOR)
    s["sensitivity_ok"] = s.scr_excess_lo > CONFIG["MIN_EXCESS"]
    s["qualifies"] = s.headroom_ok & s.sensitivity_ok
    q = s[s.qualifies].sort_values("scr_excess_lo", ascending=False)
    if q.empty:
        return s, dict(status="NO MODEL QUALIFIES", primary=None, secondary=None,
                       consequence="pivot to a model-comparison paper (planned Stage 2 gate)")
    primary = q.iloc[0].model
    other = q[q.family != FAMILY[primary]]
    if not other.empty:
        secondary, why = other.iloc[0].model, "best qualifier from a different architecture family"
    elif len(q) > 1:
        secondary, why = q.iloc[1].model, "no qualifier from another family; next best overall"
    else:
        secondary, why = None, "only one qualifier"
    return s, dict(status="SELECTED", primary=primary, secondary=secondary, secondary_rule=why,
                   ranking=q.model.tolist())


# =============================================================================
# 5. Table and figure
# =============================================================================
def fmt(m, lo, hi):
    return f"{m:0.3f} [{lo:0.3f}, {hi:0.3f}]"


def latex_table(summary, sel_df, path):
    order = [CONFIG["BASELINE"]] + CONFIG["CANDIDATES"]
    s = summary.set_index("model").loc[order]
    q = sel_df.set_index("model")
    lines = [r"\begin{tabular}{lllllll}", r"\hline",
             r"Model & Grid control acc. & Scramble excess & Frankenstein excess & Frankenstein (silhouette) & Shape bias & Qualifies \\",
             r"\hline"]
    for m, r in s.iterrows():
        qual = "baseline" if m == CONFIG["BASELINE"] else ("yes" if q.loc[m, "qualifies"] else "no")
        ex_s = "--" if m == CONFIG["BASELINE"] else fmt(r.scr_excess, r.scr_excess_lo, r.scr_excess_hi)
        ex_f = "--" if m == CONFIG["BASELINE"] else fmt(r.frk_excess, r.frk_excess_lo, r.frk_excess_hi)
        lines.append(f"{r.label} & {fmt(r.tex_cutG_acc, r.tex_cutG_lo, r.tex_cutG_hi)} & {ex_s} & {ex_f} & "
                     f"{fmt(r.sil_frank_drop, r.sil_frank_lo, r.sil_frank_hi)} & "
                     f"{fmt(r.shape_bias, r.shape_bias_lo, r.shape_bias_hi)} & {qual} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines))


# =============================================================================
# 6. Run
# =============================================================================
if RUN_MODELS:
    print("\n[2] Predictions")
    preds = run_predictions()
    print("\n[3] Measures")
    summary, acc = measures(preds)
    sel_df, selection = select(summary)
    summary.to_csv(RES / "stage2_summary.csv", index=False)
    acc.to_csv(RES / "stage2_accuracy.csv", index=False)
    sel_df.to_csv(RES / "stage2_selection_table.csv", index=False)
    latex_table(summary, sel_df, RES / "stage2_table.tex")
    C.save_json(dict(config=CONFIG, floor=FLOOR, selection=selection,
                     rule="qualify: textured cut-control lower CIs > floor and scramble-excess lower CI > 0.05; "
                          "primary = highest scramble-excess lower CI; secondary = best qualifier from another family"),
                RES / "stage2_selection.json")

    pd.set_option("display.width", 240)
    show = summary.set_index("label")
    print("\nHeadroom and configural sensitivity (textured, localizer split)")
    print(show[["tex_intact_acc", "tex_cutH_acc", "tex_cutH_lo", "tex_cutG_acc", "tex_cutG_lo",
                "scr_drop", "scr_excess", "scr_excess_lo", "scr_excess_hi", "scr_excess_cat_balanced",
                "frk_excess", "frk_excess_lo", "frk_excess_hi"]].round(3).to_string())
    print("\nPost hoc: scramble loss relative to above-chance control accuracy (not used for selection)")
    print(show[["tex_cutG_acc", "scr_rel", "scr_rel_lo", "scr_rel_hi", "scr_rel_excess", "scr_rel_excess_lo",
                "scr_rel_excess_hi"]].round(3).to_string())
    print("\nDescriptive profile")
    print(show[["sil_frank_drop", "sil_frank_lo", "sil_frank_hi", "shape_bias", "shape_bias_lo", "shape_bias_hi",
                "ref_silhouette_acc", "ref_edge_acc", "ref_sketch_acc", "ov_side_by_side", "ov_overlap_small",
                "ov_cost"]].round(3).to_string())
    print(f"  (human shape bias for reference: {HUMAN_SHAPE_BIAS:0.2f})")
    print("\n" + "=" * 78)
    for r in sel_df.itertuples():
        print(f"  {LABEL[r.model]:18s} headroom {'ok ' if r.headroom_ok else 'NO '} | "
              f"scramble excess lower CI {r.scr_excess_lo:0.3f} {'ok' if r.sensitivity_ok else 'NO'} | "
              f"{'QUALIFIES' if r.qualifies else 'does not qualify'}")
    if selection["status"] == "SELECTED":
        print(f"  PRIMARY: {LABEL[selection['primary']]}")
        print(f"  SECONDARY: {LABEL[selection['secondary']] if selection['secondary'] else 'none'} "
              f"({selection['secondary_rule']})")
    else:
        print(f"  {selection['status']}: {selection['consequence']}")
    print("=" * 78)

print("\nDone. Outputs in", RES, "and", FIGS)
