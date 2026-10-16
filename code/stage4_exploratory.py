"""
STAGE 4: Characterising the configural channels (exploratory, calibration test half)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 10 to 20 minutes on an A100).

All analyses use the calibration TEST half, which played no part in selecting the channels (the
localizer half did). Nothing touches the hold-out set. Everything here is exploratory and carries
no confirmatory claim. Channel sets come from results/stage3/lesion_sets_A1.json (Amendment A1).

At each model's primary site, three channel groups are compared:
  configural  the A1 configural set
  local       the A1 local set
  other       up to 200 channels in neither set (seeded random sample)

Analyses
  A. Effective receptive field (after Luo et al., 2016): gradient of a channel's centre unit with
     respect to the input, averaged over images; size = RMS radius of the gradient mass around its
     centroid (pixels).
  B. Attribution spread across object parts: gradient x (input - white background) attribution of a channel's spatially
     pooled response on grid-control images, summed within each of the 3 x 3 grid pieces that contain
     the object; spread = effective number of pieces (exp of entropy) / number of pieces. 1 = attribution
     spread evenly over all pieces, low = concentrated in one.
  C. Category profile: configural reliance (accuracy-based CR) per category, intact and after the
     configural lesion (zero ablation), and the animate (bear, bird, cat, dog, elephant) versus inanimate
     contrast, with object bootstrap intervals.
  D. Top images: for the four highest-I_CR configural channels, the three grid-control images that drive
     the channel most, with attribution overlays (qualitative figure).

Outputs: results/stage4/*.csv|json|tex, figures/stage4/*.pdf|png
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
from scipy import stats
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
CONFIG = dict(
    N_OTHER=200,
    N_ERF_IMAGES=64,
    N_ATTR_IMAGES=200,
    N_TOP_CHANNELS=4,
    N_TOP_IMAGES=3,
    N_BOOT=10000,
    BATCH=32,
    GRID=3,
    ANIMATE=["bear", "bird", "cat", "dog", "elephant"],
    MIN_STRENGTH=0.10,
)
if SMOKE:
    CONFIG.update(N_OTHER=20, N_ERF_IMAGES=8, N_ATTR_IMAGES=16, N_BOOT=300, BATCH=8)

C.ensure_dirs()
C.set_seed()
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage4"
FIGS = C.FIG_DIR / "stage4"
for d in [RES, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{'_smoke' if SMOKE else ''}.json"
assert SETS_PATH.exists(), "lesion_sets_A1.json not found; run Stage 3b first."
SETS = C.load_json(SETS_PATH)
H = W = C.IMG_SIZE
print(f"Stage 4 (exploratory) | sets {SETS_PATH.name} | amendment {SETS.get('amendment')} | device {DEVICE}")

# =============================================================================
# 1. Calibration test half
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("frozen_calib")
    C.unpack_stimuli("stage1")
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
tex = fc[(fc.battery == "configural_tex") & (fc.split == "test")
         & fc.condition.isin(["cutG_intact", "scrambled"])].copy()
tex = tex[~((tex.condition == "scrambled") & (tex.manip_strength < CONFIG["MIN_STRENGTH"]))]
assert not tex.path.map(C.is_holdout_path).any()
keep = sorted(set(tex[tex.condition == "cutG_intact"].source_id) & set(tex[tex.condition == "scrambled"].source_id))
if SMOKE:
    keep = keep[:24]
tex = tex[tex.source_id.isin(keep)].sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
objects = keep
oidx = {o: i for i, o in enumerate(objects)}
obj = tex.source_id.map(oidx).to_numpy()
is_grid = (tex.condition == "cutG_intact").to_numpy()
y_all = np.array(tex.category.map(C.CAT_TO_IDX).to_numpy(), dtype=np.int64)
obj_cat = tex.drop_duplicates("source_id").set_index("source_id").loc[objects, "category"].to_numpy()
grid_img = np.array([np.where(is_grid & (obj == i))[0][0] for i in range(len(objects))])
scr_avg = np.zeros((len(tex), len(objects)), np.float32)
for i in range(len(objects)):
    idx = np.where(~is_grid & (obj == i))[0]
    scr_avg[idx, i] = 1.0 / len(idx)
size = 64 if SMOKE else H
imgs = torch.from_numpy(np.stack([np.asarray(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)))
                                  for p in tex.path])).permute(0, 3, 1, 2).contiguous()
print(f"  test half: {len(objects)} objects, {len(tex)} images")

# grid geometry per object, from the object masks (same rule as the stimulus generator)
masks = {}
for f in sorted((C.STIM_LOCAL / "stage1").glob("masks_*.npz")):
    z = np.load(f)
    masks.update(dict(zip(z["source_ids"].tolist(), z["masks"])))


def grid_cells(mask):
    G = CONFIG["GRID"]
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    h, w = r1 - r0, c1 - c0
    ch, cw = min(int(np.ceil(h / G)), H // G), min(int(np.ceil(w / G)), W // G)
    R0 = int(np.clip(r0 - (G * ch - h) // 2, 0, H - G * ch))
    C0 = int(np.clip(c0 - (G * cw - w) // 2, 0, W - G * cw))
    cells = []
    for i in range(G):
        for j in range(G):
            ys, xs = slice(R0 + i * ch, R0 + (i + 1) * ch), slice(C0 + j * cw, C0 + (j + 1) * cw)
            if mask[ys, xs].sum() >= 40:
                cells.append((ys, xs))
    return cells


CELLS = {o: grid_cells(masks[o]) for o in objects if o in masks}

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def norm(x_uint8):
    return (x_uint8.to(DEVICE).float() / 255.0 - MEAN) / STD


# Attribution baseline: the blank white background every stimulus is drawn on. Gradient x (input - baseline)
# then assigns zero attribution to background pixels, so attribution mass falls on the object pieces only.
BASELINE = (torch.ones(1, 3, 1, 1, device=DEVICE) - MEAN) / STD


# =============================================================================
# 2. Models
# =============================================================================
def smoke_model(name):
    if name == "resnet50_swsl":
        import timm
        return timm.create_model("resnet50", pretrained=False).eval().to(DEVICE)
    import cornet
    return cornet.cornet_s(pretrained=False).module.eval().to(DEVICE)


class Split:
    def __init__(self, name):
        m = smoke_model(name) if SMOKE else C.load_model(name, DEVICE)
        for p in m.parameters():
            p.requires_grad_(False)
        self.name = name
        if name == "resnet50_swsl":
            stem = nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool)
            self.stages = [("stem", stem), ("layer1", m.layer1), ("layer2", m.layer2),
                           ("layer3", m.layer3), ("layer4", m.layer4)]
            self.head = lambda x: m.forward_head(x)
        else:
            self.stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.head = m.decoder
        self.index = {n: i for i, (n, _) in enumerate(self.stages)}

    def prefix(self, x, site):
        for i in range(self.index[site] + 1):
            x = self.stages[i][1](x)
        return x

    def suffix(self, a, site):
        for i in range(self.index[site] + 1, len(self.stages)):
            a = self.stages[i][1](a)
        return self.head(a)


# =============================================================================
# 3. Analyses
# =============================================================================
def erf_radius(net, site, channels):
    """RMS radius (pixels) of the input-gradient mass of each channel's centre unit."""
    sel = grid_img[np.random.default_rng(C.derive_seed("erf", net.name)).permutation(len(grid_img))[:CONFIG["N_ERF_IMAGES"]]]
    yy, xx = np.indices((size, size))
    acc = {c: np.zeros((size, size)) for c in channels}
    for b0 in range(0, len(sel), CONFIG["BATCH"]):
        x = norm(imgs[sel[b0:b0 + CONFIG["BATCH"]]]).requires_grad_(True)
        a = net.prefix(x, site)
        hc, wc = a.shape[2] // 2, a.shape[3] // 2
        for k, c in enumerate(channels):
            g, = torch.autograd.grad(a[:, c, hc, wc].sum(), x, retain_graph=(k < len(channels) - 1))
            acc[c] += g.abs().sum(dim=(0, 1)).cpu().numpy()
    out = {}
    for c, m in acc.items():
        w = m / max(m.sum(), 1e-12)
        cy, cx = (w * yy).sum(), (w * xx).sum()          # centred on the gradient centroid, not the image centre
        out[c] = float(np.sqrt((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum())) * (H / size)
    return out


def attribution_spread(net, site, channels):
    """Effective number of object pieces carrying the channel's attribution, as a fraction of pieces."""
    rng = np.random.default_rng(C.derive_seed("attr", net.name))
    objs = [i for i in rng.permutation(len(objects)) if objects[i] in CELLS and len(CELLS[objects[i]]) >= 2]
    objs = objs[:CONFIG["N_ATTR_IMAGES"]]
    sel = grid_img[objs]
    spread = {c: [] for c in channels}
    for b0 in range(0, len(sel), CONFIG["BATCH"]):
        bo = objs[b0:b0 + CONFIG["BATCH"]]
        xi = norm(imgs[sel[b0:b0 + CONFIG["BATCH"]]])
        x = xi.clone().requires_grad_(True)
        a = net.prefix(x, site)
        for k, c in enumerate(channels):
            g, = torch.autograd.grad(a[:, c].mean(dim=(1, 2)).sum(), x, retain_graph=(k < len(channels) - 1))
            att = (g * (xi - BASELINE)).abs().sum(1).cpu().numpy()    # (B, size, size)
            if size != H:
                att = np.stack([np.asarray(Image.fromarray(m.astype(np.float32)).resize((W, H))) for m in att])
            for j, oi in enumerate(bo):
                cells = CELLS[objects[oi]]
                mass = np.array([att[j][ys, xs].sum() for ys, xs in cells])
                if mass.sum() <= 0:
                    continue
                p = mass / mass.sum()
                ent = -(p[p > 0] * np.log(p[p > 0])).sum()
                spread[c].append(float(np.exp(ent) / len(cells)))
    return {c: float(np.mean(v)) if v else np.nan for c, v in spread.items()}


@torch.no_grad()
def predictions(net, site, chans):
    """16-way decisions on every test-half image, intact network and with chans zeroed."""
    out = np.zeros((2, len(tex)), bool)
    y = torch.from_numpy(y_all).to(DEVICE)
    ch = torch.as_tensor(np.asarray(chans, dtype=np.int64), device=DEVICE)
    for b0 in range(0, len(tex), CONFIG["BATCH"] * 4):
        sl = slice(b0, min(len(tex), b0 + CONFIG["BATCH"] * 4))
        a = net.prefix(norm(imgs[sl]), site)
        for k in range(2):
            aa = a.clone()
            if k == 1:
                aa[:, ch] = 0
            pred = (torch.softmax(net.suffix(aa, site).float(), 1) @ M16).argmax(1)
            out[k, sl] = (pred == y[sl]).cpu().numpy()
    return out


def cr_by_group(correct, groups, rng):
    """Accuracy-based CR (category-balanced within group) with object bootstrap, per group."""
    pg = np.zeros(len(objects))
    pg[obj[is_grid]] = correct[is_grid]
    ps = correct.astype(np.float32) @ scr_avg
    res = {}
    for gname, cats in groups.items():
        members = np.where(np.isin(obj_cat, cats))[0]
        if len(members) == 0:
            continue

        def stat(idx):
            d = pd.DataFrame(dict(c=obj_cat[idx], g=pg[idx], s=ps[idx])).groupby("c").mean()
            ag, asc = d.g.mean(), d.s.mean()
            return (ag - asc) / (ag - C.CHANCE_16) if ag >= 2 * C.CHANCE_16 else np.nan, ag, asc

        point = stat(members)
        boots = []
        for _ in range(CONFIG["N_BOOT"] // 10):
            idx = np.concatenate([rng.choice(members[obj_cat[members] == c], size=(obj_cat[members] == c).sum())
                                  for c in np.unique(obj_cat[members])])
            boots.append(stat(idx)[0])
        boots = np.array(boots, float)
        lo, hi = (np.nanquantile(boots, [0.025, 0.975]) if np.isfinite(boots).sum() > 10 else (np.nan, np.nan))
        res[gname] = dict(CR=point[0], CR_lo=float(lo), CR_hi=float(hi), A_grid=point[1], A_scr=point[2],
                          n_objects=int(len(members)))
    return res


def boot_diff(a, b, rng, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    da = rng.choice(a, size=(n, len(a))).mean(1)
    db = rng.choice(b, size=(n, len(b))).mean(1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# =============================================================================
# 4. Run
# =============================================================================
channel_rows, group_rows, cat_rows, contrast_rows, top_info = [], [], [], [], {}
scores = pd.read_csv(C.RESULTS_DIR / "stage3" / f"channel_scores_A1{'_smoke' if SMOKE else ''}.csv")
for name, mo in SETS["models"].items():
    site = mo["primary_site"]
    if site is None:
        print(f"  {name}: no primary site; skipped")
        continue
    print(f"\n[{C.MODEL_INFO[name][0]}] site {site}")
    net = Split(name)
    e = mo["sites"][site]
    conf, local = np.array(e["configural"]), np.array(e.get("local", []))
    rng = np.random.default_rng(C.derive_seed("stage4_other", name))
    rest = np.setdiff1d(np.arange(e["n_channels"]), np.union1d(conf, local))
    other = np.sort(rng.choice(rest, size=min(CONFIG["N_OTHER"], len(rest)), replace=False))
    groups = {"configural": conf, "local": local, "other": other}
    allc = np.unique(np.concatenate([g for g in groups.values() if len(g)]))
    t0 = time.time()
    erf = erf_radius(net, site, allc.tolist())
    spread = attribution_spread(net, site, allc.tolist())
    print(f"  receptive fields and attribution for {len(allc)} channels in {time.time() - t0:0.0f} s")
    sc = scores[(scores.model == name) & (scores.site == site)].set_index("channel")
    for gname, g in groups.items():
        for c in g:
            channel_rows.append(dict(model=name, site=site, group=gname, channel=int(c), erf_radius_px=erf[c],
                                     attribution_spread=spread[c], I_CR=float(sc.loc[c, "I_CR_localizer"])))
    cr_df = pd.DataFrame([r for r in channel_rows if r["model"] == name])
    rngb = np.random.default_rng(C.derive_seed("stage4_boot", name))
    for gname in groups:
        g = cr_df[cr_df.group == gname]
        for metric in ["erf_radius_px", "attribution_spread"]:
            m, lo, hi = C.bootstrap_mean_ci(g[metric], CONFIG["N_BOOT"], rng=rngb)
            group_rows.append(dict(model=name, site=site, group=gname, metric=metric, mean=m, lo=lo, hi=hi, n=len(g)))
    for gname in ["local", "other"]:
        for metric in ["erf_radius_px", "attribution_spread"]:
            a = cr_df[cr_df.group == "configural"][metric]
            b = cr_df[cr_df.group == gname][metric]
            d, lo, hi = boot_diff(a, b, rngb)
            u = stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue if len(b.dropna()) > 1 else np.nan
            contrast_rows.append(dict(model=name, metric=metric, contrast=f"configural - {gname}", diff=d, lo=lo, hi=hi,
                                      mannwhitney_p=float(u)))
    # category profile
    corr = predictions(net, site, conf)
    cat_groups = {c: [c] for c in C.CATEGORIES if c in set(obj_cat)}
    cat_groups.update(animate=CONFIG["ANIMATE"], inanimate=[c for c in C.CATEGORIES if c not in CONFIG["ANIMATE"]])
    for k, label in enumerate(["intact", "configural lesion"]):
        res = cr_by_group(corr[k], cat_groups, np.random.default_rng(C.derive_seed("stage4_cat", name, k)))
        for gname, r in res.items():
            cat_rows.append(dict(model=name, site=site, condition=label, group=gname, **r))
    # top channels and images
    top = conf[np.argsort(-sc.loc[conf, "I_CR_localizer"].to_numpy())][:CONFIG["N_TOP_CHANNELS"]]
    with torch.no_grad():
        acts = []
        for b0 in range(0, len(grid_img), 64):
            a = net.prefix(norm(imgs[grid_img[b0:b0 + 64]]), site)
            acts.append(a[:, torch.as_tensor(top, device=DEVICE)].mean(dim=(2, 3)).cpu().numpy())
        acts = np.concatenate(acts)
    panels = []
    for k, c in enumerate(top):
        best = np.argsort(-acts[:, k])[:CONFIG["N_TOP_IMAGES"]]
        for oi in best:
            xi = norm(imgs[grid_img[oi:oi + 1]])
            x = xi.clone().requires_grad_(True)
            a = net.prefix(x, site)
            g, = torch.autograd.grad(a[:, int(c)].mean(), x)
            att = (g * (xi - BASELINE)).abs().sum(1)[0].cpu().numpy()
            panels.append(dict(channel=int(c), object=objects[oi], category=obj_cat[oi],
                               image=imgs[grid_img[oi]].permute(1, 2, 0).numpy(), att=att))
    top_info[name] = dict(site=site, channels=top.tolist(), panels=panels)
    del net
    torch.cuda.empty_cache()

# =============================================================================
# 5. Save, tables, figures
# =============================================================================
ch_df, gr_df, cat_df, con_df = map(pd.DataFrame, [channel_rows, group_rows, cat_rows, contrast_rows])
sfx = "_smoke" if SMOKE else ""
ch_df.to_csv(RES / f"channel_properties{sfx}.csv", index=False)
gr_df.to_csv(RES / f"group_properties{sfx}.csv", index=False)
cat_df.to_csv(RES / f"category_profile{sfx}.csv", index=False)
con_df.to_csv(RES / f"group_contrasts{sfx}.csv", index=False)
C.save_json(dict(sets=SETS_PATH.name, config=CONFIG,
                 top_channels={m: dict(site=v["site"], channels=v["channels"],
                                       images=[dict(channel=p["channel"], object=p["object"], category=p["category"])
                                               for p in v["panels"]]) for m, v in top_info.items()}),
            RES / f"stage4_summary{sfx}.json")

# category table (LaTeX): CR intact and after the configural lesion
lines = [r"\begin{tabular}{llrrrr}", r"\hline",
         r"Model & Category & Objects & CR intact [95\% CI] & CR after configural lesion [95\% CI] & Grid accuracy after lesion \\",
         r"\hline"]
fmt = lambda m, lo, hi: "floor" if not np.isfinite(m) else (f"{m:0.2f} [{lo:0.2f}, {hi:0.2f}]" if np.isfinite(lo) else f"{m:0.2f}")
for (m, g), d in cat_df.groupby(["model", "group"], sort=False):
    i = d[d.condition == "intact"].iloc[0]
    l = d[d.condition == "configural lesion"].iloc[0]
    lines.append(f"{C.MODEL_INFO[m][0]} & {g} & {i.n_objects} & {fmt(i.CR, i.CR_lo, i.CR_hi)} & "
                 f"{fmt(l.CR, l.CR_lo, l.CR_hi)} & {l.A_grid:0.2f} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage4_category_table.tex").write_text("\n".join(lines))


def nice_bounds(lo, hi, steps=(0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100)):
    span = hi - lo
    step = next((s for s in steps if span / s <= 6), steps[-1])
    return np.floor(lo / step) * step, np.ceil(hi / step) * step, step


def fig_groups():
    """Receptive-field radius (a) and attribution spread (b) by channel group and model."""
    C.set_pub_style()
    colours = {"configural": "#2a78d6", "local": "#eb6834", "other": "#8a8a8a"}
    markers = {"configural": "o", "local": "s", "other": "D"}
    models = [m for m in SETS["models"] if m in top_info]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    for p, (ax, metric, ylab) in enumerate([(axes[0], "erf_radius_px", "Effective receptive field radius (px)"),
                                            (axes[1], "attribution_spread", "Attribution spread across pieces")]):
        d = gr_df[gr_df.metric == metric]
        xt, xl = [], []
        for i, m in enumerate(models):
            for j, g in enumerate(["configural", "local", "other"]):
                r = d[(d.model == m) & (d.group == g)]
                if r.empty or not np.isfinite(r["mean"].iloc[0]):
                    continue
                x = i * 4 + j
                ax.errorbar(x, r["mean"], yerr=[[r["mean"].iloc[0] - r["lo"].iloc[0]], [r["hi"].iloc[0] - r["mean"].iloc[0]]],
                            fmt=markers[g], ms=10, capsize=5, lw=1.8, color=colours[g],
                            label=g.capitalize() if i == 0 else None)
            xt.append(i * 4 + 1)
            xl.append(f"{C.MODEL_INFO[m][0]}\n{top_info[m]['site']}")
        vals = np.concatenate([d["lo"].to_numpy(), d["hi"].to_numpy()])
        vals = vals[np.isfinite(vals)]
        if metric == "attribution_spread":
            lo, hi, step = 0.0, 1.0, 0.2
        else:
            lo, hi, step = nice_bounds(0.0, vals.max() * 1.05 if len(vals) else 1.0)
        C.style_axes(ax, ylim=(lo, hi), ystep=step, decimals=1 if step < 1 else 0)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.set_xlim(-1, len(models) * 4 - 1)
        ax.set_ylabel(ylab)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.text(-0.12, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.3, bottom=0.22)
    C.save_figure(fig, FIGS / f"stage4_channel_groups{sfx}")
    plt.close(fig)


def fig_top():
    """Grid-control images that drive the top configural channels most, with attribution overlays."""
    C.set_pub_style()
    for m, v in top_info.items():
        P = v["panels"]
        nr, nc = CONFIG["N_TOP_CHANNELS"], CONFIG["N_TOP_IMAGES"] * 2
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.7 * nr))
        axes = np.atleast_2d(axes)
        for k, pnl in enumerate(P):
            r, cidx = divmod(k, CONFIG["N_TOP_IMAGES"])
            ax_i, ax_a = axes[r, 2 * cidx], axes[r, 2 * cidx + 1]
            ax_i.imshow(pnl["image"])
            att = pnl["att"] / max(pnl["att"].max(), 1e-12)
            ax_a.imshow(pnl["image"].mean(2), cmap="gray", vmin=0, vmax=255)
            ax_a.imshow(att, cmap="magma", alpha=0.65, vmin=0, vmax=1)
            for ax in (ax_i, ax_a):
                ax.set_xticks([])
                ax.set_yticks([])
            ax_i.set_xlabel(pnl["category"], fontsize=13)
            if cidx == 0:
                ax_i.set_ylabel(f"Channel {pnl['channel']}", fontsize=14)
        fig.subplots_adjust(wspace=0.05, hspace=0.25, right=0.88)
        top_ax, bot_ax = axes[0, -1].get_position(), axes[-1, -1].get_position()
        cax = fig.add_axes([0.9, bot_ax.y0, 0.015, top_ax.y1 - bot_ax.y0])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1)), cax=cax)
        cb.set_ticks(np.arange(0, 1.01, 0.2))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
        cb.set_label("Attribution (normalised per image)")
        cb.outline.set_linewidth(C.SPINE_LW)
        C.save_figure(fig, FIGS / f"stage4_top_channels_{m}{sfx}")
        plt.close(fig)


fig_groups()
fig_top()

pd.set_option("display.width", 220)
print("\nChannel groups (mean [95% CI])")
print(gr_df.round(3).to_string(index=False))
print("\nConfigural minus other groups")
print(con_df.round(4).to_string(index=False))
print("\nAnimate vs inanimate (CR)")
print(cat_df[cat_df.group.isin(["animate", "inanimate"])][["model", "condition", "group", "n_objects", "CR", "CR_lo",
                                                            "CR_hi", "A_grid", "A_scr"]].round(3).to_string(index=False))
print("\nDone. Outputs in", RES, "and", FIGS)
