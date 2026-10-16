"""
STAGE 1c: Frozen battery, hold-out set and lock
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU runtime). The shared module
(agnosia_common.py, now with the hold-out lock) is embedded and written to Drive.

What this stage does
  1. Freezes the stimulus design settled in Stages 1 and 1b:
       Frankenstein: 6 px horizontal cut (passed at the original setting)
       Grid scramble: 12 px gaps (chosen by the pre-specified Stage 1b rule)
     and renders it for the CALIBRATION set (the Stage 1 sources) as 'frozen_calib'.
     Stages 2 to 4 (model choice, unit selection, characterisation) use only this set.
  2. Selects a HOLD-OUT set from COCO 2017 with the same filters, excluding every image
     (not just every object) used in the calibration set, and renders it with the same
     frozen code. Textures go through the same compression history as the calibration
     set (original -> JPEG -> decode -> manipulate -> JPEG). Navon and overlapping-figure
     hold-out variants use new random seeds and hold-out sources.
  3. Fingerprints the hold-out files (SHA-256 over paths and bytes) and writes
     plan/holdout_spec.json: frozen design, selection rules, fingerprint, category
     counts, the weighting rule and the replication criteria. These are recorded BEFORE
     any model sees the hold-out images.
  4. Lock: from now on, every stage calls predict_probs16 through the shared module,
     which refuses to run any model other than the BagNet controls on hold-out images
     until plan/analysis_plan_frozen.json exists with the same fingerprint. Every hold-out
     evaluation is appended to plan/holdout_access_log.jsonl.
  5. Replication (BagNets only, allowed by the lock): repeats the local-equivalence check on
     the hold-out textured objects.

Pre-specified replication criteria (BagNet-33, hold-out, textured)
  Frankenstein (6 px) and grid scramble (12 px): control accuracy lower CI above 2 x chance,
  and the drop's 95% cluster-bootstrap CI inside +-0.05 (source-weighted, as in Stage 1).
  If either fails, the design is NOT changed; the paper reports the failure and uses the
  pre-specified fallback metric (each model's drop in excess of BagNet-33, paired per source).

Pre-specified weighting rule for later confirmatory analyses on the hold-out
  Primary means weight categories equally; categories with fewer than 20 hold-out sources
  are excluded from the primary mean and reported separately.
"""

# =============================================================================
# 0. Setup
# =============================================================================
import os
import sys
import gc
import json
import time
import shutil
import tarfile
import subprocess
import importlib
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

IN_COLAB = os.path.exists("/content") and "AGNOSIA_TEST" not in os.environ
if IN_COLAB:
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/wielandbrendel/bag-of-local-features-models.git",
                    "pycocotools"], check=True)

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

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt

RUN_MODELS = os.environ.get("AGNOSIA_NO_MODELS") is None
FAKE_HOLDOUT = os.environ.get("AGNOSIA_FAKE_HOLDOUT") is not None   # offline testing only

FROZEN = dict(
    GAP_H=6, GAP_G=12, GRID=3, N_SCRAMBLE=3, MIN_PIECE_PX=40, MIN_MANIP_STRENGTH=0.10,
    JPEG_QUALITY=95, TARGET_SIDE=176,
    NAVON_LETTERS=["E", "H", "L", "O", "T", "X"], N_NAVON=10,
    N_OVERLAP_PAIRS=320, OVERLAP_LINE_PX=2,
)
SELECTION = dict(
    N_PER_CAT=100, SPLITS=["val2017", "train2017"], MIN_AREA_FRAC=0.05, MIN_AREA_FRAC_FALLBACK=0.01,
    MIN_SIDE=96, BORDER=4, MAX_OCCLUSION=0.10, MIN_LCC=0.97, MIN_FILL=0.15,
)
REPLICATION = dict(MODELS=["bagnet9", "bagnet17", "bagnet33"], PRIMARY="bagnet33", EQ_BOUND=0.05,
                   FLOOR_MULT=2.0, N_BOOT=10000, BATCH=128,
                   calibration_reference=dict(frankenstein_gap6=0.011, scramble_gap12=0.026))
WEIGHTING = dict(primary="category-balanced", min_sources_per_category=20)
FORCE_CALIB = False
FORCE_REBUILD_HOLDOUT = False     # only honoured while the plan is not frozen

C.ensure_dirs()
C.set_seed()
C.PLAN_DIR.mkdir(parents=True, exist_ok=True)
H = W = C.IMG_SIZE
RES = C.RESULTS_DIR / "stage1c"
FIGS = C.FIG_DIR / "stage1c"
for d in [RES, FIGS, FIGS / "qc"]:
    d.mkdir(parents=True, exist_ok=True)
CALIB = "frozen_calib"
HOLD = C.HOLDOUT_STAGE
CALIB_MANIFEST = C.STIM_DRIVE / f"{CALIB}_manifest.csv"
HOLD_MANIFEST = C.STIM_DRIVE / f"{HOLD}_manifest.csv"
CALIB_CFG = C.STIM_DRIVE / f"{CALIB}_config.json"
HOLD_QC_EXCLUSIONS = C.STIM_DRIVE / "holdout_qc_exclusions.json"
COCO_IMG_LOCAL = C.RAW_LOCAL / "coco" / "images"
HOLD_IMG_DRIVE = C.RAW_DRIVE / "coco" / "images_holdout.tar"
FROZEN_HASH = C.config_hash(FROZEN)
print(f"Stage 1c | frozen design {FROZEN_HASH} | gaps H {FROZEN['GAP_H']} px, grid {FROZEN['GAP_G']} px")


# =============================================================================
# 1. Layer utilities (identical to Stage 1)
# =============================================================================
def to_png(mask, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def to_jpg(L, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(L[..., 1:])).save(path, quality=FROZEN["JPEG_QUALITY"], subsampling=0)


def jpeg_roundtrip(L):
    import io
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(L[..., 1:])).save(buf, format="JPEG", quality=FROZEN["JPEG_QUALITY"],
                                                           subsampling=0)
    rgb = np.asarray(Image.open(buf).convert("RGB"))
    out = L.copy()
    m = mask_of(L)
    out[m, 1:] = rgb[m]
    return out


def bbox(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def resize_mask(mask, out_h, out_w):
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)) > 127


def crop_to_bbox(mask):
    r0, r1, c0, c1 = bbox(mask)
    return mask[r0:r1, c0:c1]


def _paste_window(nh, nw, cy, cx):
    y0, x0 = int(round(cy - nh / 2)), int(round(cx - nw / 2))
    ys0, xs0 = max(0, -y0), max(0, -x0)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y0 + nh), min(W, x0 + nw)
    return (slice(y0c, y1c), slice(x0c, x1c)), (slice(ys0, ys0 + (y1c - y0c)), slice(xs0, xs0 + (x1c - x0c)))


def place(canvas, crop, side, cy, cx):
    h, w = crop.shape
    s = side / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    m = resize_mask(crop, nh, nw)
    dst, src = _paste_window(nh, nw, cy, cx)
    canvas[dst] |= m[src]
    return canvas


def bg_layers():
    L = np.zeros((H, W, 4), np.uint8)
    L[..., 1:] = 255
    return L


def mask_of(L):
    return L[..., 0] > 0


def clear(view):
    view[..., 0] = 0
    view[..., 1:] = 255


def layers_from_mask(mask):
    L = bg_layers()
    L[mask, 0] = 1
    L[mask, 1:] = 0
    return L


def layers_from_mask_rgb(mask, rgb):
    L = bg_layers()
    L[mask, 0] = 1
    L[mask, 1:] = rgb[mask]
    return L


def place_layers(crop_mask, crop_rgb, side, cy, cx):
    h, w = crop_mask.shape
    s = side / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    m = resize_mask(crop_mask, nh, nw)
    rgb = np.asarray(Image.fromarray(crop_rgb).resize((nw, nh), Image.BICUBIC))
    dst, src = _paste_window(nh, nw, cy, cx)
    L = bg_layers()
    sub_m = m[src]
    win = L[dst]
    win[sub_m, 0] = 1
    win[sub_m, 1:] = rgb[src][sub_m]
    return L


def outline(mask, px):
    return mask & ~ndimage.binary_erosion(mask, iterations=px)


def shift_cols_layers(L, dx):
    out = bg_layers()
    if dx > 0:
        out[:, dx:] = L[:, :-dx]
    elif dx < 0:
        out[:, :dx] = L[:, -dx:]
    else:
        out[:] = L
    return out


def paste_over(dst, src):
    m = mask_of(src)
    dst[m] = src[m]
    return dst


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


# =============================================================================
# 2. Frozen manipulations
# =============================================================================
def hsplit_control(L, gap):
    half = gap // 2
    r0, r1, _, _ = bbox(mask_of(L))
    ym = (r0 + r1) // 2
    ctrl = L.copy()
    clear(ctrl[max(0, ym - half):ym + half, :])
    return ctrl, ym


def seam_centre(part_mask, row, step, fallback):
    for k in range(12):
        r = row + k * step
        if 0 <= r < part_mask.shape[0]:
            cols = np.where(part_mask[r])[0]
            if cols.size:
                return cols.mean()
    return fallback


def frankenstein(L, which, gap):
    half = gap // 2
    ctrl, ym = hsplit_control(L, gap)
    top = ctrl.copy(); clear(top[ym:, :])
    bot = ctrl.copy(); clear(bot[:ym, :])
    mt, mb = mask_of(top), mask_of(bot)
    if mt.sum() == 0 or mb.sum() == 0:
        return None
    cent = lambda m: np.where(m)[1].mean()
    if which == "top":
        part, other, mp, mo = top, bot, mt, mb
        cp = seam_centre(mp, ym - half - 1, -1, cent(mp))
        co = seam_centre(mo, ym + half, +1, cent(mo))
    else:
        part, other, mp, mo = bot, top, mb, mt
        cp = seam_centre(mp, ym + half, +1, cent(mp))
        co = seam_centre(mo, ym - half - 1, -1, cent(mo))
    flipped = part[:, ::-1]
    dx_ideal = int(round(co - (W - 1 - cp)))
    cols = np.where(mask_of(flipped).any(axis=0))[0]
    dx = int(np.clip(dx_ideal, -cols[0], W - 1 - cols[-1]))
    moved = shift_cols_layers(flipped, dx)
    out = paste_over(other.copy(), moved)
    return dict(control=ctrl, manip=out, area_lost_frac=float(1.0 - mask_of(moved).sum() / max(1, mp.sum())),
                seam_offset_px=int(abs(dx_ideal - dx)))


def grid_geometry(mask):
    G = FROZEN["GRID"]
    r0, r1, c0, c1 = bbox(mask)
    h, w = r1 - r0, c1 - c0
    ch, cw = min(int(np.ceil(h / G)), H // G), min(int(np.ceil(w / G)), W // G)
    R0 = int(np.clip(r0 - (G * ch - h) // 2, 0, H - G * ch))
    C0 = int(np.clip(c0 - (G * cw - w) // 2, 0, W - G * cw))
    return R0, C0, ch, cw


def grid_control(L, gap):
    G, half = FROZEN["GRID"], gap // 2
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
    G = FROZEN["GRID"]
    R0, C0, ch, cw = geom
    cm = mask_of(ctrl)
    cells = [cm[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw] for i in range(G) for j in range(G)]
    nonempty = [k for k, c in enumerate(cells) if c.sum() >= FROZEN["MIN_PIECE_PX"]]
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
            best = (perm, strength, len(nonempty))
        if strength >= FROZEN["MIN_MANIP_STRENGTH"]:
            break
    return best


def apply_permutation(ctrl, geom, perm):
    G = FROZEN["GRID"]
    R0, C0, ch, cw = geom
    cells = [ctrl[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw].copy()
             for i in range(G) for j in range(G)]
    out = bg_layers()
    for k, cell in enumerate(cells):
        i, j = divmod(perm[k], G)
        paste_over(out[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw], cell)
    return out


def render_configural(stage, sources, rows):
    """sources: dicts with source_id, source_set, category, split, mask, tex (layers or None)."""
    GH, GG = FROZEN["GAP_H"], FROZEN["GAP_G"]
    for s in sources:
        sid = s["source_id"]
        L_sil = layers_from_mask(s["mask"])
        ctrlG_sil, geom = grid_control(L_sil, GG)
        cell_px = int(min(geom[2], geom[3]))
        perms = [grid_permutation(ctrlG_sil, geom, C.derive_seed("scramble", sid, v))
                 for v in range(FROZEN["N_SCRAMBLE"])]
        renders = [("configural", L_sil)]
        if s["tex"] is not None:
            renders.append(("configural_tex", s["tex"]))
        for battery, L in renders:
            base = dict(set=stage, battery=battery, source_set=s["source_set"], source_id=sid,
                        category=s["category"], split=s["split"])

            def emit(cond, LL, variant=0, **extra):
                if battery == "configural":
                    rel = f"{stage}/configural/{cond}/{sid}__v{variant}.png"
                    to_png(mask_of(LL), C.STIM_LOCAL / rel)
                else:
                    rel = f"{stage}/configural_tex/{cond}/{sid}__v{variant}.jpg"
                    to_jpg(LL, C.STIM_LOCAL / rel)
                rows.append(dict(path=rel, condition=cond, variant=variant, **base, **extra))

            emit("intact", L)
            ctrlH, _ = hsplit_control(L, GH)
            emit("cutH_intact", ctrlH, gap=GH)
            for which in ("top", "bottom"):
                f = frankenstein(L, which, GH)
                if f is None:
                    continue
                emit(f"frank_{which}", f["manip"], gap=GH, area_lost_frac=f["area_lost_frac"],
                     seam_offset_px=f["seam_offset_px"],
                     manip_strength=1 - iou(mask_of(f["control"]), mask_of(f["manip"])))
            ctrlG, _ = grid_control(L, GG)
            emit("cutG_intact", ctrlG, gap=GG, cell_px=cell_px)
            for v, r in enumerate(perms):
                if r is None:
                    continue
                perm, strength, n_pieces = r
                emit("scrambled", apply_permutation(ctrlG, geom, perm), variant=v, gap=GG, cell_px=cell_px,
                     n_pieces=n_pieces, perm=" ".join(map(str, perm)), manip_strength=strength)


# =============================================================================
# 3. Navon and overlapping figures (hold-out variants)
# =============================================================================
NAVON_BITMAPS = {
    "E": ["11111", "10000", "11110", "10000", "11111"], "H": ["10001", "10001", "11111", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "11111"], "O": ["01110", "10001", "10001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100"], "X": ["10001", "01010", "00100", "01010", "10001"],
}
NAVON_CELLS = {L: [(r, c) for r, row in enumerate(b) for c, ch in enumerate(row) if ch == "1"]
               for L, b in NAVON_BITMAPS.items()}


def render_navon(cells, local, rng, square=False):
    spacing, size = int(rng.integers(32, 39)), int(rng.integers(22, 29))
    oy, ox = rng.integers(-8, 9, size=2)
    img = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"), size)
    y_start, x_start = H / 2 - 2 * spacing + oy, W / 2 - 2 * spacing + ox
    for r, c in cells:
        cy, cx = y_start + r * spacing, x_start + c * spacing
        if square:
            s = 0.3 * size
            dr.rectangle([cx - s, cy - s, cx + s, cy + s], fill=0)
        else:
            dr.text((cx, cy), local, fill=0, font=font, anchor="mm")
    return img, dict(spacing=spacing, font_px=size)


def build_navon(stage, rows):
    letters, n = FROZEN["NAVON_LETTERS"], FROZEN["N_NAVON"]

    def save(img, rel, **meta):
        p = C.STIM_LOCAL / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        rows.append(dict(path=rel, set=stage, battery="navon", source_set="navon", category="", split=stage, **meta))

    for g in letters:
        for l in letters:
            cond = "navon_congruent" if g == l else "navon_incongruent"
            for v in range(n):
                img, meta = render_navon(NAVON_CELLS[g], l, np.random.default_rng(C.derive_seed(stage, "navon", g, l, v)))
                save(img, f"{stage}/navon/{cond}/G{g}_L{l}__v{v}.png", condition=cond, source_id=f"navon_G{g}_L{l}",
                     variant=v, global_letter=g, local_letter=l, n_elements=len(NAVON_CELLS[g]), **meta)
        for v in range(n):
            img, meta = render_navon(NAVON_CELLS[g], "", np.random.default_rng(C.derive_seed(stage, "navon_g", g, v)), square=True)
            save(img, f"{stage}/navon/navon_global_only/G{g}__v{v}.png", condition="navon_global_only",
                 source_id=f"navon_G{g}", variant=v, global_letter=g, local_letter="",
                 n_elements=len(NAVON_CELLS[g]), **meta)
    counts = [len(NAVON_CELLS[g]) for g in letters]
    all_cells = [(r, c) for r in range(5) for c in range(5)]
    for l in letters:
        for v in range(n):
            rng = np.random.default_rng(C.derive_seed(stage, "navon_l", l, v))
            k = counts[v % len(counts)]
            cells = [all_cells[i] for i in rng.choice(25, size=k, replace=False)]
            img, meta = render_navon(cells, l, rng)
            save(img, f"{stage}/navon/navon_local_only/L{l}__v{v}.png", condition="navon_local_only",
                 source_id=f"navon_L{l}", variant=v, global_letter="", local_letter=l, n_elements=k, **meta)


def build_overlap(stage, sources, rows):
    px = FROZEN["OVERLAP_LINE_PX"]
    crops = {s["source_id"]: crop_to_bbox(s["mask"]) for s in sources}
    meta = {s["source_id"]: s for s in sources}

    def save(mask, rel, **extra):
        to_png(mask, C.STIM_LOCAL / rel)
        rows.append(dict(path=rel, set=stage, battery="overlap", **extra))

    for s in sources:
        canvas = place(np.zeros((H, W), bool), crops[s["source_id"]], 170, H / 2, W / 2)
        save(outline(canvas, px), f"{stage}/overlap/outline_single/{s['source_id']}.png", condition="outline_single",
             source_set=s["source_set"], source_id=s["source_id"], category=s["category"], split=s["split"], variant=0)
    rng = np.random.default_rng(C.derive_seed(stage, "overlap_pairs"))
    ids = [s["source_id"] for s in sources]
    seen, made, guard = set(), 0, 0
    while made < FROZEN["N_OVERLAP_PAIRS"] and guard < 200000 and len(ids) > 1:
        guard += 1
        a, b = rng.choice(ids, size=2, replace=False)
        if meta[a]["category"] == meta[b]["category"] or (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        pid = f"pair{made:04d}"
        j = rng.integers(-10, 11, size=4)
        variants = {"overlap_large": [(170, H / 2 + j[0], W / 2 + j[1]), (170, H / 2 + j[2], W / 2 + j[3])],
                    "overlap_small": [(100, H / 2, W / 2 - 25), (100, H / 2, W / 2 + 25)],
                    "side_by_side": [(100, H / 2, 58), (100, H / 2, 166)]}
        for cond, specs in variants.items():
            ink, single = np.zeros((H, W), bool), 0
            for sid, (side, cy, cx) in zip((a, b), specs):
                o = outline(place(np.zeros((H, W), bool), crops[sid], side, cy, cx), px)
                single += o.sum()
                ink |= o
            save(ink, f"{stage}/overlap/{cond}/{pid}.png", condition=cond, source_set="pair", source_id=pid,
                 category=meta[a]["category"], category_b=meta[b]["category"], source_a=a, source_b=b,
                 split=meta[a]["split"], variant=0, ink_px=int(ink.sum()),
                 ink_overlap_frac=float(1 - ink.sum() / max(1, single)))
        made += 1


# =============================================================================
# 4. Calibration sources (from Stage 1) and the frozen calibration battery
# =============================================================================
print("\n[1] Calibration set")
if IN_COLAB:
    C.unpack_stimuli("stage1")
S1 = C.STIM_LOCAL / "stage1"
man1 = pd.read_csv(C.STIM_DRIVE / "stage1_manifest.csv", low_memory=False)
mask_by_id = {}
for f in sorted(S1.glob("masks_*.npz")):
    z = np.load(f)
    mask_by_id.update(dict(zip(z["source_ids"].tolist(), z["masks"])))
sil_intact = man1[(man1.battery == "configural") & (man1.condition == "intact")].set_index("source_id")
tex_intact = man1[(man1.battery == "configural_tex") & (man1.condition == "intact")].set_index("source_id")
calib_ids = [sid for sid in sil_intact.index if sid in mask_by_id]
calib_coco_ann = {int(sid.split("_")[1]) for sid in calib_ids if sid.startswith("coco_")}


def calib_sources():
    out = []
    for sid in calib_ids:
        mask = mask_by_id[sid]
        tex = None
        if sid in tex_intact.index:
            rgb = np.asarray(Image.open(C.STIM_LOCAL / tex_intact.loc[sid, "path"]).convert("RGB"))
            tex = layers_from_mask_rgb(mask, rgb)
        out.append(dict(source_id=sid, source_set=sil_intact.loc[sid, "source_set"],
                        category=sil_intact.loc[sid, "category"], split=sil_intact.loc[sid, "split"],
                        mask=mask, tex=tex))
    return out


def build_calib():
    if (C.STIM_LOCAL / CALIB).exists():
        shutil.rmtree(C.STIM_LOCAL / CALIB)
    rows = []
    t0 = time.time()
    render_configural(CALIB, calib_sources(), rows)
    man = pd.DataFrame(rows)
    man.to_csv(CALIB_MANIFEST, index=False)
    C.save_json(dict(frozen=FROZEN, hash=FROZEN_HASH, n_images=len(man)), CALIB_CFG)
    print(f"  rendered {len(man)} calibration images in {time.time() - t0:0.0f} s")
    if IN_COLAB:
        C.pack_stimuli(CALIB)
    return man


cached = (not FORCE_CALIB and CALIB_CFG.exists() and CALIB_MANIFEST.exists()
          and C.load_json(CALIB_CFG).get("hash") == FROZEN_HASH
          and (C.stimuli_archive_path(CALIB).exists() or not IN_COLAB))
if cached:
    print("  reusing frozen calibration battery")
    if IN_COLAB:
        C.unpack_stimuli(CALIB)
    man_calib = pd.read_csv(CALIB_MANIFEST, low_memory=False)
else:
    man_calib = build_calib()
print(man_calib.groupby(["battery", "condition"]).size().to_string())


# =============================================================================
# 5. Hold-out selection
# =============================================================================
def select_holdout(excluded_qc):
    from pycocotools.coco import COCO
    ann_dir = C.get_coco_annotations()
    n_target = SELECTION["N_PER_CAT"]
    selected = {c: [] for c in C.CATEGORIES}
    used_images = set()
    taken = set()
    log = {c: dict(considered=0, calibration_image=0, small=0, border=0, occluded=0, fragmented=0,
                   sparse=0, excluded_qc=0, duplicate_image=0) for c in C.CATEGORIES}
    n_calib_images = 0
    for thr in [SELECTION["MIN_AREA_FRAC"], SELECTION["MIN_AREA_FRAC_FALLBACK"]]:
        for split in SELECTION["SPLITS"]:
            if all(len(v) >= n_target for v in selected.values()):
                break
            t0 = time.time()
            coco = COCO(str(ann_dir / f"instances_{split}.json"))
            calib_images = {coco.anns[a]["image_id"] for a in calib_coco_ann if a in coco.anns}
            n_calib_images += len(calib_images) if thr == SELECTION["MIN_AREA_FRAC"] else 0
            print(f"  COCO {split} loaded in {time.time() - t0:0.0f} s | area threshold {thr} | "
                  f"{len(calib_images)} calibration images excluded")
            for cat in C.CATEGORIES:
                if len(selected[cat]) >= n_target:
                    continue
                cat_id = coco.getCatIds(catNms=[cat])[0]
                anns = coco.loadAnns(coco.getAnnIds(catIds=[cat_id], iscrowd=False))
                order = np.random.default_rng(C.derive_seed("holdout", split, cat)).permutation(len(anns))
                lg = log[cat]
                for i in order:
                    if len(selected[cat]) >= n_target:
                        break
                    ann = anns[i]
                    sid = f"coco_{ann['id']}"
                    if sid in taken or ann["id"] in calib_coco_ann:
                        continue
                    lg["considered"] += 1
                    if ann["image_id"] in calib_images:
                        lg["calibration_image"] += 1
                        continue
                    if sid in excluded_qc:
                        lg["excluded_qc"] += 1
                        continue
                    if ann["image_id"] in used_images:
                        lg["duplicate_image"] += 1
                        continue
                    img = coco.imgs[ann["image_id"]]
                    ih, iw = img["height"], img["width"]
                    x, y, w, h = ann["bbox"]
                    if ann["area"] < thr * ih * iw or max(w, h) < SELECTION["MIN_SIDE"]:
                        lg["small"] += 1
                        continue
                    b = SELECTION["BORDER"]
                    if x <= b or y <= b or x + w >= iw - b or y + h >= ih - b:
                        lg["border"] += 1
                        continue
                    occ = 0.0
                    for other in coco.imgToAnns[ann["image_id"]]:
                        if other["id"] == ann["id"]:
                            continue
                        ox, oy, ow, oh = other["bbox"]
                        occ = max(occ, max(0.0, min(x + w, ox + ow) - max(x, ox)) *
                                  max(0.0, min(y + h, oy + oh) - max(y, oy)) / (w * h))
                    if occ > SELECTION["MAX_OCCLUSION"]:
                        lg["occluded"] += 1
                        continue
                    m = coco.annToMask(ann).astype(bool)
                    lab, n = ndimage.label(m)
                    if n == 0:
                        lg["fragmented"] += 1
                        continue
                    sizes = ndimage.sum(m, lab, range(1, n + 1))
                    if sizes.max() / sizes.sum() < SELECTION["MIN_LCC"]:
                        lg["fragmented"] += 1
                        continue
                    m = lab == (np.argmax(sizes) + 1)
                    r0, r1, c0, c1 = bbox(m)
                    crop = m[r0:r1, c0:c1]
                    if crop.mean() < SELECTION["MIN_FILL"]:
                        lg["sparse"] += 1
                        continue
                    selected[cat].append(dict(source_id=sid, category=cat, split_coco=split,
                                              file_name=img["file_name"], image_id=int(ann["image_id"]),
                                              crop_bits=np.packbits(crop), crop_shape=crop.shape,
                                              img_bbox=(r0, r1, c0, c1), area_threshold=thr))
                    used_images.add(ann["image_id"])
                    taken.add(sid)
            del coco
            gc.collect()
    counts = {c: len(v) for c, v in selected.items()}
    print(f"  hold-out instances: {sum(counts.values())} | per category {counts}")
    C.save_json(dict(counts=counts, rejections=log), RES / "holdout_selection_log.json")
    return [r for c in C.CATEGORIES for r in selected[c]]


def fetch_images(records):
    COCO_IMG_LOCAL.mkdir(parents=True, exist_ok=True)
    if HOLD_IMG_DRIVE.exists():
        with tarfile.open(HOLD_IMG_DRIVE) as t:
            t.extractall(COCO_IMG_LOCAL)
    todo = sorted({(r["split_coco"], r["file_name"]) for r in records
                   if not (COCO_IMG_LOCAL / r["split_coco"] / r["file_name"]).exists()})

    def get(item):
        split, fn = item
        dst = COCO_IMG_LOCAL / split / fn
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/{split}/{fn}", dst)
                return True
            except Exception:
                time.sleep(1 + attempt)
        return False

    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(32) as ex:
            ok = list(ex.map(get, todo))
        print(f"  downloaded {sum(ok)} / {len(todo)} images in {time.time() - t0:0.0f} s")
    if IN_COLAB:
        tmp = Path("/tmp/images_holdout.tar")
        with tarfile.open(tmp, "w") as t:
            for r in records:
                p = COCO_IMG_LOCAL / r["split_coco"] / r["file_name"]
                if p.exists():
                    t.add(p, arcname=f"{r['split_coco']}/{r['file_name']}")
        HOLD_IMG_DRIVE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp, HOLD_IMG_DRIVE)
        tmp.unlink()


def holdout_sources(excluded_qc):
    if FAKE_HOLDOUT:
        out = []
        for sid in [s for s in calib_ids if s.startswith("geirhos_")]:
            mask = mask_by_id[sid]
            rng = np.random.default_rng(C.derive_seed("fakehold", sid))
            noise = np.clip(ndimage.gaussian_filter(rng.random((H, W, 3)) * 255, sigma=(3, 3, 0)), 0, 255).astype(np.uint8)
            out.append(dict(source_id=f"fake_{sid}", source_set="coco", category=sil_intact.loc[sid, "category"],
                            split=HOLD, mask=mask, tex=jpeg_roundtrip(layers_from_mask_rgb(mask, noise)),
                            image_id=-1, area_threshold=np.nan))
        return out
    records = select_holdout(excluded_qc)
    fetch_images(records)
    out, n_missing = [], 0
    for r in records:
        crop = np.unpackbits(r["crop_bits"])[:np.prod(r["crop_shape"])].reshape(r["crop_shape"]).astype(bool)
        p = COCO_IMG_LOCAL / r["split_coco"] / r["file_name"]
        tex = None
        if p.exists():
            try:
                img = np.asarray(Image.open(p).convert("RGB"))
                r0, r1, c0, c1 = r["img_bbox"]
                rgb = img[r0:r1, c0:c1]
                if rgb.shape[:2] == crop.shape:
                    tex = jpeg_roundtrip(place_layers(crop, rgb, FROZEN["TARGET_SIDE"], H / 2, W / 2))
            except Exception:
                tex = None
        if tex is not None:
            mask = mask_of(tex)
        else:
            n_missing += 1
            mask = place(np.zeros((H, W), bool), crop, FROZEN["TARGET_SIDE"], H / 2, W / 2)
        out.append(dict(source_id=r["source_id"], source_set="coco", category=r["category"], split=HOLD,
                        mask=mask, tex=tex, image_id=r["image_id"], area_threshold=r["area_threshold"]))
    print(f"  hold-out sources: {len(out)} | texture unavailable for {n_missing}")
    return out


def qc_sheets(sources):
    for cat in C.CATEGORIES:
        items = [s for s in sources if s["category"] == cat]
        if not items:
            continue
        ncol = 10
        nrow = int(np.ceil(len(items) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.8))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, s in zip(axes.ravel(), items):
            ax.imshow(s["tex"][..., 1:] if s["tex"] is not None else ~s["mask"], cmap=None if s["tex"] is not None else "gray")
            ax.text(0.5, -0.06, s["source_id"].replace("coco_", ""), transform=ax.transAxes, ha="center", va="top", fontsize=7)
        fig.savefig(FIGS / "qc" / f"holdout_{cat}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 6. Build, fingerprint and specify the hold-out (before any model sees it)
# =============================================================================
def build_holdout():
    excluded = set(C.load_json(HOLD_QC_EXCLUSIONS)) if HOLD_QC_EXCLUSIONS.exists() else set()
    if (C.STIM_LOCAL / HOLD).exists():
        shutil.rmtree(C.STIM_LOCAL / HOLD)
    sources = holdout_sources(excluded)
    qc_sheets(sources)
    np.savez_compressed(RES / "holdout_masks.npz", source_ids=np.array([s["source_id"] for s in sources]),
                        masks=np.stack([s["mask"] for s in sources]))
    rows = []
    t0 = time.time()
    render_configural(HOLD, sources, rows)
    build_navon(HOLD, rows)
    build_overlap(HOLD, sources, rows)
    man = pd.DataFrame(rows)
    print(f"  rendered {len(man)} hold-out images in {time.time() - t0:0.0f} s")
    man.to_csv(HOLD_MANIFEST, index=False)
    fp = C.fingerprint_files(C.STIM_LOCAL, man.path.tolist())
    tex = man[(man.battery == "configural_tex") & (man.condition == "intact")]
    per_cat = tex.groupby("category").size().reindex(C.CATEGORIES, fill_value=0).astype(int).to_dict()
    spec = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), fingerprint=fp, n_images=len(man),
        manifest_sha256=C.hashlib.sha256(HOLD_MANIFEST.read_bytes()).hexdigest(),
        frozen_design=FROZEN, frozen_design_hash=FROZEN_HASH, selection=SELECTION,
        exclusion="every COCO image containing a calibration object is excluded; no image appears twice",
        n_textured_sources_per_category=per_cat,
        categories_below_min=[c for c, n in per_cat.items() if n < WEIGHTING["min_sources_per_category"]],
        weighting_rule=WEIGHTING, replication_criteria=REPLICATION,
        design_history=[
            "v1: binary silhouettes only; local-only control (BagNet-33) at chance, so the behavioural check was uninformative",
            "v2: textured COCO objects added; Frankenstein passed; grid scramble at 6 px failed the +-0.05 bound (BagNet-33 drop 0.054 [0.041, 0.068])",
            "1b: gap-width sensitivity with a pre-specified rule selected 12 px (BagNet-33 drop 0.026 [0.015, 0.038])",
            "1c: design frozen; hold-out generated with disjoint images; lock enabled",
        ])
    C.save_json(spec, C.HOLDOUT_SPEC)
    if IN_COLAB:
        print("  archiving hold-out to Drive ...")
        C.pack_stimuli(HOLD)
    return man, spec


print("\n[2] Hold-out set")
spec_exists = C.HOLDOUT_SPEC.exists() and HOLD_MANIFEST.exists()
plan_frozen = C.FROZEN_PLAN.exists()
if spec_exists and not (FORCE_REBUILD_HOLDOUT and not plan_frozen):
    print("  hold-out already exists; verifying fingerprint")
    if IN_COLAB:
        C.unpack_stimuli(HOLD)
    man_hold = pd.read_csv(HOLD_MANIFEST, low_memory=False)
    spec = C.load_json(C.HOLDOUT_SPEC)
    fp = C.fingerprint_files(C.STIM_LOCAL, man_hold.path.tolist())
    assert fp == spec["fingerprint"], "Hold-out files differ from the recorded fingerprint."
    print("  fingerprint verified")
else:
    if spec_exists and plan_frozen:
        raise RuntimeError("Plan is frozen; the hold-out cannot be rebuilt.")
    man_hold, spec = build_holdout()
print(man_hold.groupby(["battery", "condition"]).size().to_string())
print(f"\n  HOLD-OUT FINGERPRINT: {spec['fingerprint']}")
print(f"  created {spec['created']} | textured sources per category: {spec['n_textured_sources_per_category']}")
if spec["categories_below_min"]:
    print(f"  below {WEIGHTING['min_sources_per_category']} sources (reported separately): {spec['categories_below_min']}")
print(f"  lock status for study models: {C.holdout_status()[1]}")


# =============================================================================
# 7. Replication of the local-equivalence check (BagNets only)
# =============================================================================
def run_replication():
    import torch
    device = C.get_device()
    ev = man_hold[man_hold.battery == "configural_tex"].reset_index(drop=True)
    paths = [str(C.STIM_LOCAL / p) for p in ev.path]
    preds = {}
    for name in REPLICATION["MODELS"]:
        out = RES / f"holdout_predictions_{name}.csv"
        if out.exists():
            df = pd.read_csv(out)
            if df.path.tolist() == ev.path.tolist():
                preds[name] = df
                print(f"  {name}: cached")
                continue
        t0 = time.time()
        model = C.load_model(name, device)
        p16 = C.predict_probs16(model, paths, device, batch_size=REPLICATION["BATCH"], model_name=name)
        del model
        torch.cuda.empty_cache()
        df = ev[["path", "condition", "variant", "source_id", "category", "manip_strength"]].copy()
        y = df.category.map(C.CAT_TO_IDX).to_numpy()
        df["correct"] = (p16.argmax(1) == y).astype(int)
        df["p_true"] = p16[np.arange(len(y)), y]
        df.to_csv(out, index=False)
        preds[name] = df
        print(f"  {name}: {time.time() - t0:0.0f} s")
    return preds


def replication_analysis(preds):
    rng = np.random.default_rng(C.derive_seed("replication"))
    floor = REPLICATION["FLOOR_MULT"] * C.CHANCE_16
    rows = []
    for name, df in preds.items():
        d = df[~(df.condition.isin(["frank_top", "frank_bottom", "scrambled"])
                 & (df.manip_strength < FROZEN["MIN_MANIP_STRENGTH"]))].copy()
        d["cond"] = d.condition.replace({"frank_top": "frankenstein", "frank_bottom": "frankenstein"})
        ps = d.groupby(["source_id", "category", "cond"]).correct.mean().reset_index()
        wide = ps.pivot(index="source_id", columns="cond", values="correct")
        cat = ps.drop_duplicates("source_id").set_index("source_id").category
        for fam, ctrl, manip in [("frankenstein", "cutH_intact", "frankenstein"), ("scramble", "cutG_intact", "scrambled")]:
            both = wide[[ctrl, manip]].dropna().index
            dd = (wide.loc[both, ctrl] - wide.loc[both, manip]).to_numpy()
            m, lo, hi = C.bootstrap_mean_ci(dd, REPLICATION["N_BOOT"], rng=rng)
            cm, clo, chi = C.bootstrap_mean_ci(wide.loc[both, ctrl], REPLICATION["N_BOOT"], rng=rng)
            c = cat.loc[both]
            keep = c.map(c.value_counts()) >= WEIGHTING["min_sources_per_category"]
            bal = float(pd.Series(dd[keep.to_numpy()]).groupby(c[keep].to_numpy()).mean().mean()) if keep.any() else np.nan
            floor_ok = bool(clo > floor)
            rows.append(dict(model=name, family=fam, n_sources=len(both), ctrl_acc=cm, ctrl_lo=clo, ctrl_hi=chi,
                             drop=m, drop_lo=lo, drop_hi=hi, drop_category_balanced=bal, floor_ok=floor_ok,
                             decision=C.equivalence_decision(lo, hi, m, REPLICATION["EQ_BOUND"]) if floor_ok else "floor"))
    return pd.DataFrame(rows)


if RUN_MODELS:
    print("\n[3] Replication on hold-out (BagNet controls only)")
    preds = run_replication()
    rep = replication_analysis(preds)
    rep.to_csv(RES / "holdout_replication.csv", index=False)
    prim = rep[rep.model == REPLICATION["PRIMARY"]].set_index("family")
    status = {f: ("REPLICATED" if prim.loc[f, "decision"] == "equivalent" else "NOT REPLICATED") for f in prim.index}
    overall = "REPLICATED" if all(v == "REPLICATED" for v in status.values()) else "NOT REPLICATED (fallback metric applies)"
    C.save_json(dict(by_family=status, overall=overall, table=rep.to_dict(orient="records"),
                     calibration_reference=REPLICATION["calibration_reference"]), RES / "holdout_replication.json")
    pd.set_option("display.width", 220)
    print(rep[["model", "family", "n_sources", "ctrl_acc", "ctrl_lo", "drop", "drop_lo", "drop_hi",
               "drop_category_balanced", "decision"]].round(3).to_string(index=False))
    print("\n" + "=" * 78)
    ref = REPLICATION["calibration_reference"]
    print(f"  Calibration reference (BagNet-33): Frankenstein 0.011, scramble at 12 px 0.026")
    for f, v in status.items():
        print(f"  Hold-out {f:13s} {v}")
    print(f"  STAGE 1c: {overall}")
    print(f"  Study models on hold-out: {C.holdout_status()[1].upper()}")
    print("=" * 78)

print("\nDone. Hold-out spec at", C.HOLDOUT_SPEC)
