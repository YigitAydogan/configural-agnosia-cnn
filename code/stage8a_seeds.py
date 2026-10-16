"""
STAGE 8a: Training-seed replicates of the progressive-lesioning analysis (exploratory)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 1.5 to 2 hours on an A100). Every track
(model x retraining regime x lesion order x training seed) is saved when it finishes; rerunning skips
finished tracks, so a disconnect costs at most one track.

Why: Stage 7 trained each track once. Comparing its two runs suggested retraining noise of about 0.05 in
configural reliance, and two random-order tracks diverged numerically. This stage repeats Stage 7 with
five training seeds per track and a numerically stable loss, so that seed-to-seed variation is measured
rather than guessed.

Identical to Stage 7 except:
  - Seeds: 5 training seeds per track (seeds change data order, augmentation and nothing else); the
    lesion orders (configural, 3 random, none) are the same for every seed.
  - Loss computed in log space: log of the 16-way probability = log-mean-exp of the category's ImageNet
    log-probabilities minus the log-sum over categories. Mathematically the same quantity as Stage 7's,
    without the underflow that made the probability ratio unstable. Gradient clipping at 1 is kept; an
    update is skipped if the loss or any gradient is not finite.
  - Same divergence rule as Stage 7 (training-set accuracy below 0.5 after any retraining: track flagged
    and left out, reported).
Summary per model, regime and contrast (final step, after retraining): the mean over seeds of
CR(configural) - CR(no lesion) and CR(configural) - mean CR(random), with a category-stratified object
bootstrap of that seed mean (objects resampled jointly for all seeds), plus the per-seed values and their
range. The same for grid-control accuracy.

Questions are those of Stage 7 (Q1 to Q3); this stage asks whether the answers hold across seeds.
Exploratory; nothing here changes the confirmatory results.

Outputs: results/stage8a/*.csv|json|tex, figures/stage8a/*.pdf|png
"""

# =============================================================================
# 0. Setup
# =============================================================================
import os
import sys
import copy
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
import matplotlib.pyplot as plt
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
sfx = ("_smoke" if SMOKE else "") + ("_fake" if os.environ.get("AGNOSIA_FAKE") else "")
SETS_SHA_REGISTERED = "9bb05eb1ee8729786246ff80f23575225c88f3271c4c6756a92525500c2fb904"
FROZEN = C.load_json(C.FROZEN_PLAN)
PLAN = FROZEN["plan"]
ok, why = C.holdout_status()
assert ok, f"Hold-out locked: {why}"
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{'_smoke' if SMOKE else ''}.json"
sets_sha = hashlib.sha256(SETS_PATH.read_bytes()).hexdigest()
if not SMOKE:
    assert sets_sha == SETS_SHA_REGISTERED, "lesion_sets_A1.json differs from the version published in v1.2."
SETS = C.load_json(SETS_PATH)
SCORES = pd.read_csv(C.RESULTS_DIR / "stage3" / f"channel_scores_A1{'_smoke' if SMOKE else ''}.csv")

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    STEPS=[0.01, 0.02, 0.05, 0.10, 0.20],      # then the A1 configural set size
    N_RANDOM_ORDERS=3,
    REGIMES=["readout", "downstream"],
    EPOCHS=3, TRAIN_BATCH=64, GRAD_CLIP=1.0, DIVERGED_TRAIN_ACC=0.5, SEEDS=[1, 2, 3, 4, 5],
    LR={"readout": 1e-3, "downstream_body": 3e-5, "downstream_head": 1e-3},
    SHIFT_PX=16,
    N_BOOT=10000, EVAL_BATCH=128,
)
if SMOKE:
    CONFIG.update(N_RANDOM_ORDERS=1, EPOCHS=1, TRAIN_BATCH=16, N_BOOT=200, EVAL_BATCH=32, MIN_OBJECTS_PER_CATEGORY=2,
                  STEPS=[0.05, 0.20], SEEDS=[1, 2])

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage8a"
CKPT = RES / "checkpoints"
FIGS = C.FIG_DIR / "stage8a"
for d in [RES, CKPT, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
CFG_TAG = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:8]
print(f"Stage 8a (exploratory, seed replicates) | lesion sets {sets_sha[:12]} | device {DEVICE}")

# =============================================================================
# 1. Images
# =============================================================================
if IN_COLAB:
    for st in ["frozen_calib", C.HOLDOUT_STAGE]:
        C.unpack_stimuli(st)
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
train = fc[(fc.battery == "configural_tex") & (fc.condition == "intact")].reset_index(drop=True)
if SMOKE:
    train = train.iloc[:64]
assert not train.path.map(C.is_holdout_path).any()

d = hm[(hm.battery == "configural_tex") & hm.condition.isin(["cutG_intact", "scrambled", "intact"])].copy()
d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_STRENGTH"]))]
d = d[~d.category.isin(CONFIG["EXCLUDED"])]
keep = (set(d[d.condition == "cutG_intact"].source_id) & set(d[d.condition == "scrambled"].source_id)
        & set(d[d.condition == "intact"].source_id))
d = d[d.source_id.isin(keep)]
n_per = d.drop_duplicates("source_id").category.value_counts()
d = d[~d.category.isin(n_per[n_per < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
d = d.sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
OBJECTS = sorted(d.source_id.unique())
oi = {o: i for i, o in enumerate(OBJECTS)}
obj = d.source_id.map(oi).to_numpy()
CAT = d.drop_duplicates("source_id").set_index("source_id").loc[OBJECTS, "category"].to_numpy()
Y_TEST = d.category.map(C.CAT_TO_IDX).to_numpy()
GRID = np.array([np.where((d.condition == "cutG_intact").to_numpy() & (obj == i))[0][0] for i in range(len(OBJECTS))])
INTACT = np.array([np.where((d.condition == "intact").to_numpy() & (obj == i))[0][0] for i in range(len(OBJECTS))])
SCR = np.zeros((len(d), len(OBJECTS)), np.float32)
for i in range(len(OBJECTS)):
    idx = np.where((d.condition == "scrambled").to_numpy() & (obj == i))[0]
    SCR[idx, i] = 1.0 / len(idx)

size = 64 if SMOKE else C.IMG_SIZE
load = lambda paths: torch.from_numpy(np.stack([np.array(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)),
                                                         dtype=np.uint8) for p in paths])).permute(0, 3, 1, 2).contiguous()
t0 = time.time()
X_TRAIN, Y_TRAIN = load(train.path), torch.as_tensor(np.array(train.category.map(C.CAT_TO_IDX).to_numpy(), dtype=np.int64))
X_TEST = load(d.path)
HO_PATHS = d.path.tolist()
print(f"  training: {len(train)} calibration intact objects | test: {len(OBJECTS)} hold-out objects, {len(d)} images "
      f"| loaded in {time.time() - t0:0.0f} s")

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def norm(x):
    return (x.to(DEVICE, non_blocking=True).float() / 255.0 - MEAN) / STD


def augment(x):
    """Random horizontal flip and translation (white padding), per image, on uint8 tensors."""
    n, _, h, w = x.shape
    s = CONFIG["SHIFT_PX"] * h // C.IMG_SIZE
    pad = torch.full((n, 3, h + 2 * s, w + 2 * s), 255, dtype=x.dtype)
    pad[:, :, s:s + h, s:s + w] = x
    out = torch.empty_like(x)
    oy, ox = np.random.randint(0, 2 * s + 1, n), np.random.randint(0, 2 * s + 1, n)
    flip = np.random.rand(n) < 0.5
    for i in range(n):
        im = pad[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w]
        out[i] = im.flip(-1) if flip[i] else im
    return out


# =============================================================================
# 2. Models with a lesion at the site and trainable downstream layers
# =============================================================================
def smoke_model(name):
    if name == "resnet50_swsl":
        import timm
        return timm.create_model("resnet50", pretrained=False).eval().to(DEVICE)
    import cornet
    return cornet.cornet_s(pretrained=False).module.eval().to(DEVICE)


class Net(nn.Module):
    def __init__(self, name, site):
        super().__init__()
        m = smoke_model(name) if SMOKE else C.load_model(name, DEVICE)
        self.name = name
        if name == "resnet50_swsl":
            stages = [("stem", nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool)), ("layer1", m.layer1),
                      ("layer2", m.layer2), ("layer3", m.layer3), ("layer4", m.layer4)]
            self.pool = lambda x: m.forward_head(x, pre_logits=True)
            self.fc = m.fc
        else:
            stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.pool = lambda x: m.decoder.flatten(m.decoder.avgpool(x))
            self.fc = m.decoder.linear
        k = [n for n, _ in stages].index(site)
        self.up = nn.ModuleList([b for _, b in stages[:k + 1]])        # frozen, up to and including the site
        self.down = nn.ModuleList([b for _, b in stages[k + 1:]])      # trainable in the downstream regime
        self.m = m
        self.lesion = torch.zeros(0, dtype=torch.long, device=DEVICE)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def site(self, x):
        with torch.no_grad():
            for b in self.up:
                x = b(x)
            if len(self.lesion):
                x = x.clone()
                x[:, self.lesion] = 0
        return x

    def forward(self, x):
        a = self.site(x)
        for b in self.down:
            a = b(a)
        return self.fc(self.pool(a))


LOGMASK = torch.where(M16.T > 0, torch.zeros_like(M16.T), torch.full_like(M16.T, float("-inf")))   # (16, 1000)
LOG_N = torch.log((M16.T > 0).sum(1).float())


def log_p16n(logits):
    """log of the renormalised 16-way probability, computed in log space (no underflow)."""
    lsm = torch.log_softmax(logits.float(), 1)
    lp = torch.logsumexp(lsm[:, None, :] + LOGMASK[None], dim=2) - LOG_N[None]
    return lp - torch.logsumexp(lp, dim=1, keepdim=True)


def retrain(net, regime, seed):
    """Brief retraining on calibration intact objects; BatchNorm statistics stay frozen (eval mode)."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if regime == "readout":
        params = [{"params": list(net.fc.parameters()), "lr": CONFIG["LR"]["readout"]}]
    else:
        params = [{"params": [p for b in net.down for p in b.parameters()], "lr": CONFIG["LR"]["downstream_body"]},
                  {"params": list(net.fc.parameters()), "lr": CONFIG["LR"]["downstream_head"]}]
    for g in params:
        for p in g["params"]:
            p.requires_grad_(True)
    opt = torch.optim.Adam(params)
    n = len(X_TRAIN)
    losses = []
    for ep in range(CONFIG["EPOCHS"]):
        perm = torch.randperm(n)
        for b0 in range(0, n, CONFIG["TRAIN_BATCH"]):
            idx = perm[b0:b0 + CONFIG["TRAIN_BATCH"]]
            x = norm(augment(X_TRAIN[idx]))
            y = Y_TRAIN[idx].to(DEVICE)
            loss = -log_p16n(net(x)).gather(1, y[:, None]).mean()
            opt.zero_grad()
            if not torch.isfinite(loss):
                continue                                          # numerical safeguard: skip the update
            loss.backward()
            plist = [p for g in params for p in g["params"]]
            if not all(torch.isfinite(p.grad).all() for p in plist if p.grad is not None):
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(plist, CONFIG["GRAD_CLIP"])
            opt.step()
            losses.append(loss.item())
    for g in params:
        for p in g["params"]:
            p.requires_grad_(False)
    return float(np.mean(losses[-max(1, n // CONFIG["TRAIN_BATCH"]):])) if losses else float("nan")


FAKE = SMOKE and os.environ.get("AGNOSIA_FAKE") is not None    # offline test of the statistics only
_fake_rng = np.random.default_rng(0)


@torch.no_grad()
def evaluate(net):
    """Correct 16-way decisions on every hold-out test image, plus training-set accuracy."""
    if FAKE:
        k = len(net.lesion)
        pg, ps = 0.7 - 0.001 * k, 0.35 + 0.0015 * k
        cond = d.condition.to_numpy()
        p = np.where(cond == "scrambled", ps, pg)
        return _fake_rng.random(len(d)) < p, (0.3 if (k > 0 and _fake_rng.random() < 0.03) else 0.8)
    out = np.zeros(len(d), bool)
    yt = torch.as_tensor(Y_TEST, device=DEVICE)
    for b0 in range(0, len(d), CONFIG["EVAL_BATCH"]):
        pred = log_p16n(net(norm(X_TEST[b0:b0 + CONFIG["EVAL_BATCH"]]))).argmax(1)
        out[b0:b0 + len(pred)] = (pred == yt[b0:b0 + len(pred)]).cpu().numpy()
    tr = []
    for b0 in range(0, len(X_TRAIN), CONFIG["EVAL_BATCH"]):
        pred = log_p16n(net(norm(X_TRAIN[b0:b0 + CONFIG["EVAL_BATCH"]]))).argmax(1).cpu()
        tr.append((pred == Y_TRAIN[b0:b0 + len(pred)]).numpy())
    return out, float(np.concatenate(tr).mean())


# =============================================================================
# 3. Tracks (checkpointed)
# =============================================================================
def orders_for(name):
    site = SETS["models"][name]["primary_site"]
    e = SETS["models"][name]["sites"][site]
    n = e["n_channels"]
    sc = SCORES[(SCORES.model == name) & (SCORES.site == site)].set_index("channel").loc[np.arange(n), "I_CR_localizer"]
    conf_order = np.argsort(-np.nan_to_num(sc.to_numpy(), nan=-np.inf))
    sizes = [max(1, int(round(f * n))) for f in CONFIG["STEPS"]] + [len(e["configural"])]
    sizes = sorted(set(sizes))
    orders = {"configural": conf_order, "none": np.array([], int)}
    for r in range(CONFIG["N_RANDOM_ORDERS"]):
        orders[f"random {r + 1}"] = np.random.default_rng(C.derive_seed("stage7_random_order", name, r)).permutation(n)
    return site, n, sizes, orders, len(e["configural"])


ALL_ROWS = []
STEP_INFO = {}
for name in CONFIG["MODELS"]:
    site, n_ch, sizes, orders, a1 = orders_for(name)
    STEP_INFO[name] = dict(site=site, n_channels=n_ch, sizes=sizes, a1_size=a1)
    print(f"\n[{C.MODEL_INFO[name][0]}] site {site} ({n_ch} channels) | steps {sizes} channels (A1 set {a1})")
    logged = False
    for seed, regime, (order_name, order) in [(s_, r_, o_) for s_ in CONFIG["SEEDS"] for r_ in CONFIG["REGIMES"]
                                               for o_ in orders.items()]:
        if True:
            ck = CKPT / f"{name}_{regime}_{order_name.replace(' ', '_')}_seed{seed}_{CFG_TAG}{sfx}.npz"
            if ck.exists():
                z = np.load(ck, allow_pickle=True)
                ALL_ROWS += list(z["rows"])
                print(f"  seed {seed} {regime:<10} {order_name:<10} loaded from checkpoint")
                continue
            if not logged:
                C.check_holdout_access(HO_PATHS, name)
                logged = True
            t0 = time.time()
            net = Net(name, site)
            rows = []
            base_corr, base_tr = evaluate(net)
            rows.append(dict(model=name, seed=seed, regime=regime, order=order_name, step=0, n_lesioned=0, phase="before",
                             correct=base_corr, train_acc=base_tr, loss=np.nan))
            for si, k in enumerate(sizes):
                if order_name != "none":
                    net.lesion = torch.as_tensor(np.sort(order[:k]), device=DEVICE)
                corr_b, tr_b = evaluate(net)
                loss = 0.0 if FAKE else retrain(net, regime, C.derive_seed("stage8a_train", name, regime, order_name, si, seed))
                corr_a, tr_a = evaluate(net)
                nl = 0 if order_name == "none" else k
                rows.append(dict(model=name, seed=seed, regime=regime, order=order_name, step=si + 1, n_lesioned=nl,
                                 phase="before", correct=corr_b, train_acc=tr_b, loss=np.nan))
                rows.append(dict(model=name, seed=seed, regime=regime, order=order_name, step=si + 1, n_lesioned=nl,
                                 phase="after", correct=corr_a, train_acc=tr_a, loss=loss))
            np.savez_compressed(ck, rows=np.array(rows, dtype=object))
            ALL_ROWS += rows
            print(f"  seed {seed} {regime:<10} {order_name:<10} done in {(time.time() - t0) / 60:0.1f} min | final training accuracy "
                  f"{rows[-1]['train_acc']:0.3f}")
            del net
            torch.cuda.empty_cache()

# =============================================================================
# 4. Statistics
# =============================================================================
rng = np.random.default_rng(C.derive_seed("stage8a_boot"))
cats = sorted(set(CAT))
W0 = np.zeros(len(OBJECTS), np.float32)
W = np.zeros((CONFIG["N_BOOT"], len(OBJECTS)), np.float32)
for c in cats:
    g = np.where(CAT == c)[0]
    s = 1.0 / (len(g) * len(cats))
    W0[g] = s
    dr = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
    cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
    np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), dr.ravel()), 1)
    W[:, g] = cnt * s


def cr(ag, asc):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ag - asc) / (ag - C.CHANCE_16)


table, boot = [], {}
for r in ALL_ROWS:
    c = r["correct"]
    g, s_, i_ = c[GRID].astype(np.float32), c.astype(np.float32) @ SCR, c[INTACT].astype(np.float32)
    ag, asc, ai = W0 @ g, W0 @ s_, W0 @ i_
    key = (r["model"], r["seed"], r["regime"], r["order"], r["step"], r["phase"])
    need_boot = r["phase"] == "after" or r["step"] == 0
    AG = W @ g if need_boot else None
    lo_g = float(np.quantile(AG, 0.025)) if need_boot else np.nan
    floor = bool(need_boot and lo_g < 2 * C.CHANCE_16)
    if need_boot:
        boot[key] = dict(CR=cr(AG, W @ s_), AG=AG, floor=floor)
    table.append(dict(model=r["model"], seed=r["seed"], regime=r["regime"], order=r["order"], step=r["step"],
                      n_lesioned=r["n_lesioned"], phase=r["phase"], A_intact=float(ai), A_grid=float(ag), A_grid_lo=lo_g,
                      A_scr=float(asc), CR=np.nan if floor else float(cr(ag, asc)), floor=floor, train_acc=r["train_acc"]))
tab = pd.DataFrame(table)
div = tab[(tab.phase == "after") & (tab.train_acc < CONFIG["DIVERGED_TRAIN_ACC"])][["model", "seed", "regime", "order"]]
DIVERGED = {tuple(x) for x in div.drop_duplicates().itertuples(index=False)}
tab["diverged"] = [(m, s_, r_, o) in DIVERGED for m, s_, r_, o in zip(tab.model, tab.seed, tab.regime, tab.order)]
print(f"\n  Diverged tracks (left out): {sorted(DIVERGED) if DIVERGED else 'none'}")

per_seed, pooled = [], []
for name in CONFIG["MODELS"]:
    last = len(STEP_INFO[name]["sizes"])
    rnd_orders = [f"random {i + 1}" for i in range(CONFIG["N_RANDOM_ORDERS"])]
    for regime in CONFIG["REGIMES"]:
        for ref in ["no lesion", "mean random"]:
            for meas in ["CR", "AG"]:
                pts, bts = [], []
                for seed in CONFIG["SEEDS"]:
                    k = lambda o: (name, seed, regime, o, last, "after")
                    ok = lambda o: k(o) in boot and (name, seed, regime, o) not in DIVERGED
                    if not ok("configural") or (meas == "CR" and boot[k("configural")]["floor"]):
                        continue
                    bc = boot[k("configural")]
                    if ref == "no lesion":
                        if not ok("none") or (meas == "CR" and boot[k("none")]["floor"]):
                            continue
                        other_b = boot[k("none")][meas]
                        use = ["none"]
                    else:
                        use = [o for o in rnd_orders if ok(o) and not (meas == "CR" and boot[k(o)]["floor"])]
                        if not use:
                            continue
                        other_b = np.mean([boot[k(o)][meas] for o in use], 0)
                    col = "CR" if meas == "CR" else "A_grid"
                    t_ = tab[(tab.model == name) & (tab.seed == seed) & (tab.regime == regime) & (tab.step == last)
                             & (tab.phase == "after")].set_index("order")
                    pt = float(t_.loc["configural", col] - t_.loc[use, col].mean())
                    pts.append(pt)
                    bts.append(bc[meas] - other_b)
                    per_seed.append(dict(model=name, regime=regime, contrast=f"configural - {ref}",
                                         measure="CR" if meas == "CR" else "grid accuracy", seed=seed, estimate=pt,
                                         n_reference_tracks=len(use)))
                if not pts:
                    continue
                mb = np.nanmean(np.stack(bts, 0), 0)                 # seed mean, same object resample for every seed
                pooled.append(dict(model=name, regime=regime, contrast=f"configural - {ref}",
                                   measure="CR" if meas == "CR" else "grid accuracy", n_seeds=len(pts),
                                   seed_mean=float(np.mean(pts)), lo=float(np.nanquantile(mb, 0.025)),
                                   hi=float(np.nanquantile(mb, 0.975)), seed_sd=float(np.std(pts, ddof=1)) if len(pts) > 1 else np.nan,
                                   seed_min=float(np.min(pts)), seed_max=float(np.max(pts)),
                                   seeds_same_sign=int(np.sum(np.sign(pts) == np.sign(np.mean(pts))))))
ps, pl = pd.DataFrame(per_seed), pd.DataFrame(pooled)

tab.to_csv(RES / f"seed_steps{sfx}.csv", index=False)
ps.to_csv(RES / f"per_seed_contrasts{sfx}.csv", index=False)
pl.to_csv(RES / f"pooled_contrasts{sfx}.csv", index=False)
C.save_json(dict(config=CONFIG, steps=STEP_INFO, lesion_sets_sha256=sets_sha, diverged=sorted(map(list, DIVERGED)),
                 run_at=time.strftime("%Y-%m-%d %H:%M:%S")), RES / f"stage8a_summary{sfx}.json")

lines = [r"\begin{tabular}{lllrrrl}", r"\hline",
         r"Model & Retraining & Contrast & Seeds & Seed mean [95\% CI] & Seed range & Same sign \\", r"\hline"]
for r in pl[pl.measure == "CR"].itertuples():
    lines.append(f"{C.MODEL_INFO[r.model][0]} & {r.regime} & CR, {r.contrast} & {r.n_seeds} & "
                 f"{r.seed_mean:0.3f} [{r.lo:0.3f}, {r.hi:0.3f}] & {r.seed_min:0.3f} to {r.seed_max:0.3f} & "
                 f"{r.seeds_same_sign}/{r.n_seeds} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage8a_table.tex").write_text("\n".join(lines))


# =============================================================================
# 5. Figure: CR after each retraining step, mean over seeds (seed range shaded for the configural order)
# =============================================================================
def tidy(ax, dec):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{dec}f}"))


def figure():
    C.set_pub_style()
    fig, axes = plt.subplots(len(CONFIG["MODELS"]), len(CONFIG["REGIMES"]),
                             figsize=(8.5 * len(CONFIG["REGIMES"]), 6.8 * len(CONFIG["MODELS"])))
    axes = np.atleast_2d(axes)
    t = tab[~tab.diverged & ((tab.phase == "after") | (tab.step == 0))]
    vals = t.CR.dropna().to_numpy()
    lo = min(0.0, np.floor(vals.min() / 0.2) * 0.2) if len(vals) else 0.0
    hi = max(1.0, np.ceil(vals.max() / 0.2) * 0.2) if len(vals) else 1.0
    for r, name in enumerate(CONFIG["MODELS"]):
        sizes = STEP_INFO[name]["sizes"]
        pct = [0.0] + [100 * k / STEP_INFO[name]["n_channels"] for k in sizes]
        xs = np.arange(len(pct))
        for c, regime in enumerate(CONFIG["REGIMES"]):
            ax = axes[r, c]
            g = t[(t.model == name) & (t.regime == regime)]

            def per_seed_curves(pred):
                out = []
                for seed in CONFIG["SEEDS"]:
                    s_ = g[(g.seed == seed) & pred(g)].groupby("step").CR.mean().reindex(range(len(pct)))
                    out.append(s_.to_numpy())
                return np.array(out, float)

            rnd = per_seed_curves(lambda x: x.order.str.startswith("random"))
            non = per_seed_curves(lambda x: x.order == "none")
            cfg = per_seed_curves(lambda x: x.order == "configural")
            ax.plot(xs, np.nanmean(rnd, 0), color="#8a8a8a", marker="D", ms=7, lw=2, label="Random orders (mean over seeds)")
            ax.plot(xs, np.nanmean(non, 0), color="#000000", ls="--", marker="o", mfc="white", ms=8, lw=2,
                    label="No lesion, same retraining")
            ax.fill_between(xs, np.nanmin(cfg, 0), np.nanmax(cfg, 0), color="#2a78d6", alpha=0.2, lw=0,
                            label="Configural, range over seeds")
            ax.plot(xs, np.nanmean(cfg, 0), color="#2a78d6", marker="o", ms=9, lw=2.5, label="Configural (mean over seeds)")
            C.style_axes(ax, ylim=(lo, hi), ystep=0.2, decimals=1)
            tidy(ax, 1)
            ax.set_xticks(xs)
            ax.set_xticklabels(["0"] + [f"{int(round(p))}" if p >= 0.95 else f"{p:0.1f}" for p in pct[1:]])
            ax.set_xlim(-0.3, len(xs) - 0.7)
            ax.set_xlabel(f"Channels lesioned at {STEP_INFO[name]['site']} (%), {regime} retraining")
            ax.set_ylabel(f"Configural reliance\n{C.MODEL_INFO[name][0]}")
            ax.yaxis.set_label_coords(-0.13, 0.5)
            ax.text(-0.13, 1.03, "abcd"[r * len(CONFIG["REGIMES"]) + c], transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    C.outline_legend(leg)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.14)
    C.save_figure(fig, FIGS / f"stage8a_seeds{sfx}")
    plt.close(fig)


figure()

pd.set_option("display.width", 240)
show = lambda df: df.round(4).astype(object).where(df.notna(), "").to_string(index=False)
print("\nFinal step after retraining, every seed (configural, no lesion, random mean)")
fin = tab[(tab.phase == "after") & ~tab.diverged]
fin = fin[fin.step == fin.groupby("model").step.transform("max")]
fin = fin.assign(order_group=np.where(fin.order.str.startswith("random"), "random (mean)", fin.order))
print(show(fin.groupby(["model", "regime", "order_group", "seed"])[["A_intact", "A_grid", "A_scr", "CR"]].mean().reset_index()))
print("\nPer-seed contrasts")
print(show(ps))
print("\nPooled over seeds (object bootstrap of the seed mean; seed range and sign agreement)")
print(show(pl))
print("\nDone. Outputs in", RES, "and", FIGS)
