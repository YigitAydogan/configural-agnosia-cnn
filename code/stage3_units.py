"""
STAGE 3: Unit localisation (calibration set only)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run. GPU required; an A100 is best
(about 1 to 2 hours in total). Progress is saved to Drive after every site and half, so if
Colab disconnects, re-run the cell and it continues where it stopped.

Follows the frozen analysis plan (plan/analysis_plan_frozen.json), section "Unit localisation"
and the set definitions of "Lesion protocol". Nothing here touches the hold-out set.

For each model (ResNet-50 SWSL, CORnet-S) and site (layer1-4; V1, V2, V4, IT):
  1. Single-channel zero ablation of every channel, on the textured grid controls (12 px) and
     scrambles of the calibration localizer half and, separately, the test half.
       I_conf(c) = [P_grid - P_grid(-c)] - [P_scr - P_scr(-c)]   (P: probability of the true category)
       local(c)  = P_scr - P_scr(-c)
  2. Reliability: Spearman correlation of I_conf between halves, Spearman-Brown corrected;
     a site is lesioned only if it reaches 0.7.
  3. Sets (localizer half):
       configural: one-sided t-test of I_conf across objects, Benjamini-Hochberg q = 0.05,
                   bounded to 1-40% of channels
       local:      channels with I_conf <= median, ranked by local score, size chosen so damage
                   (drop in category-balanced grid-control accuracy) is closest to the configural set's
       selective:  paired t-test, spatially averaged response grid > scrambled, BH q = 0.05
  4. Primary site per model: reliable site with the largest summed I_conf over its configural set.
  5. Control sets at the primary site: 100 size-matched random sets; 100 damage-matched random
     sets (from up to 10,000, within 0.02 of the configural damage, else the 100 closest).
     Dose-response sets (1-40%) with 20 size-matched random sets each; every other reliable site
     with its configural set and 20 size-matched random sets.
  6. Channel means over calibration images, for the mean-ablation robustness analysis.
  7. Exploratory pilot on the calibration test half: CR after the configural, local and 20 random
     lesions at the primary site. Not a confirmatory result.

Outputs on Drive: results/stage3/lesion_sets.json (read by later stages), per-channel scores,
a summary table (LaTeX) and a split-half reliability figure.
"""

# =============================================================================
# 0. Setup
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
from scipy import stats
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

assert C.FROZEN_PLAN.exists(), "The analysis plan must be frozen before Stage 3."
PLAN = C.load_json(C.FROZEN_PLAN)["plan"]
PLAN_SHA = C.load_json(C.FROZEN_PLAN)["plan_sha256"]

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    SITES=PLAN["models"]["sites"],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    Q=0.05,
    MIN_FRAC=0.01,
    MAX_FRAC=0.40,
    REL_THRESHOLD=0.7,
    N_RANDOM_SIZE=PLAN["lesions"]["random_size_matched"]["n_sets"],
    N_DAMAGE=PLAN["lesions"]["random_damage_matched"]["n_sets"],
    DAMAGE_POOL=PLAN["lesions"]["random_damage_matched"]["pool"],
    DAMAGE_TOL=PLAN["lesions"]["random_damage_matched"]["tolerance"],
    DOSE_FRACS=PLAN["lesions"]["dose_response"]["fractions"],
    DOSE_RANDOM=PLAN["lesions"]["dose_response"]["random_sets_per_size"],
    OTHER_SITE_RANDOM=PLAN["lesions"]["other_sites"]["random_sets"],
    PILOT_RANDOM=20,
    BATCH=128,
    MAX_SUFFIX_IMAGES=1024,
    MAX_ELEMENTS=4e8,
)
SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
if SMOKE:
    CONFIG.update(N_RANDOM_SIZE=10, N_DAMAGE=10, DAMAGE_POOL=300, DOSE_RANDOM=3, OTHER_SITE_RANDOM=3,
                  PILOT_RANDOM=3, REL_THRESHOLD=-1.0, BATCH=32,
                  SITES={m: v[-2:] for m, v in CONFIG["SITES"].items()})

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage3"
CKPT = RES / "checkpoints"
FIGS = C.FIG_DIR / "stage3"
for d in [RES, CKPT, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
print(f"Stage 3 | plan {PLAN_SHA[:12]} | device {DEVICE} | models {CONFIG['MODELS']}")


# =============================================================================
# 1. Images (calibration only, textured grid controls and scrambles)
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("frozen_calib")
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
tex = fc[(fc.battery == "configural_tex") & fc.condition.isin(["cutG_intact", "scrambled"])].copy()
tex = tex[~((tex.condition == "scrambled") & (tex.manip_strength < CONFIG["MIN_STRENGTH"]))]
assert not tex.path.map(C.is_holdout_path).any()


class Half:
    """Images and object bookkeeping for one half of the calibration set."""

    def __init__(self, name):
        d = tex[tex.split == name]
        has_grid = set(d[d.condition == "cutG_intact"].source_id)
        has_scr = set(d[d.condition == "scrambled"].source_id)
        keep = sorted(has_grid & has_scr)
        if SMOKE:
            keep = keep[:24]
        d = d[d.source_id.isin(keep)].sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
        self.name, self.df = name, d
        self.objects = keep
        oidx = {o: i for i, o in enumerate(keep)}
        self.obj = d.source_id.map(oidx).to_numpy()
        self.is_grid = (d.condition == "cutG_intact").to_numpy()
        self.y = d.category.map(C.CAT_TO_IDX).to_numpy()
        cats = d.drop_duplicates("source_id").set_index("source_id").loc[keep, "category"]
        self.obj_cat = cats.map(C.CAT_TO_IDX).to_numpy()
        self.grid_img = np.array([np.where(self.is_grid & (self.obj == i))[0][0] for i in range(len(keep))])
        # averaging matrix over each object's scrambles: (N_images, n_objects)
        S = np.zeros((len(d), len(keep)), np.float32)
        for i in range(len(keep)):
            idx = np.where(~self.is_grid & (self.obj == i))[0]
            S[idx, i] = 1.0 / len(idx)
        self.scr_avg = S
        t0 = time.time()
        size = 64 if SMOKE else C.IMG_SIZE
        self.imgs = torch.from_numpy(np.stack([np.asarray(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)))
                                               for p in d.path])).permute(0, 3, 1, 2).contiguous()
        print(f"  {name}: {len(keep)} objects, {len(d)} images ({self.is_grid.sum()} grid, "
              f"{(~self.is_grid).sum()} scrambled) loaded in {time.time() - t0:0.0f} s")

    def balanced(self, per_obj):
        """Category-balanced mean of a per-object quantity."""
        s = pd.Series(per_obj).groupby(self.obj_cat).mean()
        return float(s.mean())


print("\n[1] Calibration images")
HALVES = {h: Half(h) for h in ["localizer", "test"]}

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def norm(x_uint8):
    return (x_uint8.to(DEVICE, non_blocking=True).float() / 255.0 - MEAN) / STD


# =============================================================================
# 2. Model split into stages so activations can be cached at a site
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
        self.name = name
        if name == "resnet50_swsl":
            stem = nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool)
            self.stages = [("stem", stem), ("layer1", m.layer1), ("layer2", m.layer2),
                           ("layer3", m.layer3), ("layer4", m.layer4)]
            self.head = lambda x: m.forward_head(x)
        elif name == "cornet_s":
            self.stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.head = m.decoder
        else:
            raise ValueError(name)
        self.model = m
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
        return self.head(a)


@torch.no_grad()
def p16(logits):
    return torch.softmax(logits.float(), dim=1) @ M16


# =============================================================================
# 3. Single-channel ablation scores (checkpointed per model, site and half)
# =============================================================================
def channel_scores(net, site, half):
    """D[c, i]: drop in true-category probability on image i when channel c is zeroed. Each chunk of
    ablated copies is run in the same call as an unablated copy, so numerical differences between
    batch shapes never appear as lesion effects."""
    ck = CKPT / f"{net.name}_{site}_{half.name}{'_smoke' if SMOKE else ''}.npz"
    if ck.exists():
        z = np.load(ck)
        if z["D"].shape[1] == len(half.df):
            return z["D"], z["A"], z["sum_act"], int(z["n_act"])
    N = len(half.df)
    y = torch.from_numpy(half.y).to(DEVICE)
    with torch.no_grad():
        probe = net.prefix(norm(half.imgs[:2]), site)
    Cn, Hs, Ws = probe.shape[1:]
    B = CONFIG["BATCH"]
    K = int(max(1, min(CONFIG["MAX_SUFFIX_IMAGES"] // B, CONFIG["MAX_ELEMENTS"] // (B * Cn * Hs * Ws))))
    D = np.zeros((Cn, N), np.float32)
    A = np.zeros((N, Cn), np.float32)
    sum_act = np.zeros(Cn, np.float64)
    t0 = time.time()
    for b0 in range(0, N, B):
        sl = slice(b0, min(N, b0 + B))
        yb = y[sl]
        a = net.prefix(norm(half.imgs[sl]), site)
        A[sl] = a.float().mean(dim=(2, 3)).cpu().numpy()
        sum_act += a.float().mean(dim=(2, 3)).sum(0).double().cpu().numpy()
        nb = a.shape[0]
        for c0 in range(0, Cn, K):
            chans = list(range(c0, min(Cn, c0 + K)))
            aa = a.unsqueeze(0).repeat(len(chans) + 1, 1, 1, 1, 1)      # copy 0 stays intact
            for k, c in enumerate(chans):
                aa[k + 1, :, c] = 0
            out = p16(net.suffix(aa.flatten(0, 1), site))
            pt = out.gather(1, yb.repeat(len(chans) + 1)[:, None])[:, 0].view(len(chans) + 1, nb)
            D[c0:c0 + len(chans), sl] = (pt[0:1] - pt[1:]).cpu().numpy()
        done = min(N, b0 + B)
        el = time.time() - t0
        print(f"\r    {net.name} {site} {half.name}: {done}/{N} images | {el / 60:0.1f} min | "
              f"ETA {el / done * (N - done) / 60:0.1f} min", end="")
    print()
    np.savez_compressed(ck, D=D, A=A, sum_act=sum_act, n_act=N)
    return D, A, sum_act, N


def object_level(D, half):
    """Per-channel, per-object drops on grid controls and scrambles."""
    d_img = D                                               # (C, N): drop caused by ablating c
    d_grid = d_img[:, half.grid_img]                        # (C, n_obj)
    d_scr = d_img @ half.scr_avg                            # (C, n_obj)
    return d_grid, d_scr


def bh(p):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def site_scores(net, site):
    res = {}
    for hname, half in HALVES.items():
        D, A, sum_act, n_act = channel_scores(net, site, half)
        d_grid, d_scr = object_level(D, half)
        I_obj = d_grid - d_scr
        res[hname] = dict(I=I_obj.mean(1), I_obj=I_obj, local=d_scr.mean(1), A=A,
                          sum_act=sum_act, n_act=n_act)
    loc = res["localizer"]
    t, p = stats.ttest_1samp(loc["I_obj"], 0.0, axis=1, alternative="greater")
    p = np.nan_to_num(p, nan=1.0)
    # selectivity: spatially averaged response, grid control vs mean over scrambles, per object
    h = HALVES["localizer"]
    act_grid = loc["A"][h.grid_img]                         # (n_obj, C)
    act_scr = h.scr_avg.T @ loc["A"]                        # (n_obj, C)
    ts, ps = stats.ttest_rel(act_grid, act_scr, axis=0, alternative="greater")
    ps = np.nan_to_num(ps, nan=1.0)
    r, _ = stats.spearmanr(loc["I"], res["test"]["I"])
    sb = 2 * r / (1 + r) if r > -1 else np.nan
    total = res["localizer"]["n_act"] + res["test"]["n_act"]
    mean_act = (res["localizer"]["sum_act"] + res["test"]["sum_act"]) / total
    return dict(I_loc=loc["I"], I_test=res["test"]["I"], local=loc["local"], t=t, p=p, q=bh(p),
                t_sel=ts, q_sel=bh(ps), spearman=float(r), sb=float(sb), mean_act=mean_act,
                n_channels=len(loc["I"]))


# =============================================================================
# 4. Group-lesion evaluation (correctness per image for many channel sets)
# =============================================================================
@torch.no_grad()
def eval_sets(net, site, half, sets, img_mask=None, mode="zero", mean_act=None):
    """Returns a (n_sets + 1, N_sel) boolean array of correct 16-way decisions; row 0 = no lesion.
    Several sets share one suffix call (copy 0 of each call is the unlesioned network)."""
    idx = np.arange(len(half.df)) if img_mask is None else np.where(img_mask)[0]
    y_all = torch.from_numpy(half.y).to(DEVICE)
    out = np.zeros((len(sets) + 1, len(idx)), bool)
    B = CONFIG["BATCH"]
    mvals = None if mean_act is None else torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
    for b0 in range(0, len(idx), B):
        sel = idx[b0:b0 + B]
        a = net.prefix(norm(half.imgs[sel]), site)
        nb = a.shape[0]
        K = int(max(1, min(CONFIG["MAX_SUFFIX_IMAGES"] // nb - 1, CONFIG["MAX_ELEMENTS"] // a.numel() - 1)))
        yb = y_all[sel]
        for s0 in range(0, max(1, len(sets)), K):
            chunk = sets[s0:s0 + K]
            aa = a.unsqueeze(0).repeat(len(chunk) + 1, 1, 1, 1, 1)
            for k, chans in enumerate(chunk):
                ch = torch.as_tensor(np.asarray(chans, dtype=np.int64), device=DEVICE)
                if len(ch) == 0:
                    continue
                if mode == "zero":
                    aa[k + 1][:, ch] = 0
                else:
                    aa[k + 1][:, ch] = mvals[ch].to(aa.dtype).view(1, -1, 1, 1)
            pred = p16(net.suffix(aa.flatten(0, 1), site)).argmax(1).view(len(chunk) + 1, nb)
            corr = (pred == yb[None, :]).cpu().numpy()
            if s0 == 0:
                out[0, b0:b0 + nb] = corr[0]
            out[1 + s0:1 + s0 + len(chunk), b0:b0 + nb] = corr[1:]
    return out


def damages(net, site, sets):
    h = HALVES["localizer"]
    rows = eval_sets(net, site, h, sets, img_mask=h.is_grid)
    # eval_sets returns rows in grid-image order; map them to object order
    obj_of_grid = h.obj[h.is_grid]
    accs = []
    for r in rows:
        per_obj = np.zeros(len(h.objects))
        per_obj[obj_of_grid] = r
        accs.append(h.balanced(per_obj))
    accs = np.array(accs)
    return accs[0] - accs[1:], accs[0]


def cr(correct_row, half):
    """Configural reliance from one correctness row over all images of a half."""
    per_obj_grid = np.zeros(len(half.objects))
    per_obj_grid[half.obj[half.is_grid]] = correct_row[half.is_grid]
    per_obj_scr = correct_row.astype(np.float32) @ half.scr_avg
    ag, asc = half.balanced(per_obj_grid), half.balanced(per_obj_scr)
    floor = 2 * C.CHANCE_16                                # plan: no CR below twice chance
    value = (ag - asc) / (ag - C.CHANCE_16) if ag >= floor else np.nan
    return value, ag, asc


# =============================================================================
# 5. Sets per site
# =============================================================================
def configural_set(sc):
    n = sc["n_channels"]
    lo, hi = max(1, int(round(CONFIG["MIN_FRAC"] * n))), int(round(CONFIG["MAX_FRAC"] * n))
    order = np.argsort(-sc["I_loc"])
    sig = np.where(sc["q"] < CONFIG["Q"])[0]
    k = len(sig)
    if k < lo:
        chosen, rule = order[:lo], f"fewer than 1% significant ({k}); top 1% taken"
    elif k > hi:
        chosen, rule = order[:hi], f"more than 40% significant ({k}); top 40% taken"
    else:
        chosen, rule = sig[np.argsort(-sc["I_loc"][sig])], f"{k} FDR-significant channels"
    return np.sort(chosen), rule, k


def random_sets(n_channels, k, n_sets, *key):
    rng = np.random.default_rng(C.derive_seed(*key))
    return [np.sort(rng.choice(n_channels, size=k, replace=False)) for _ in range(n_sets)]


def local_set(net, site, sc, target_damage):
    elig = np.where(sc["I_loc"] <= np.median(sc["I_loc"]))[0]
    ranked = elig[np.argsort(-sc["local"][elig])]
    cache = {}

    def dmg(k):
        if k not in cache:
            cache[k] = float(damages(net, site, [ranked[:k]])[0][0])
        return cache[k]

    grid = sorted(set(np.unique(np.round(np.geomspace(1, len(ranked), 25)).astype(int)).tolist()))
    best = min(grid, key=lambda k: abs(dmg(k) - target_damage))
    i = grid.index(best)
    lo = grid[max(0, i - 1)]
    hi = grid[min(len(grid) - 1, i + 1)]
    span = list(range(lo, hi + 1))
    if len(span) > 40:
        span = sorted(set(np.linspace(lo, hi, 40).astype(int).tolist()))
    best = min(span, key=lambda k: abs(dmg(k) - target_damage))
    return np.sort(ranked[:best]), dmg(best)


def damage_matched(net, site, k, target, n_channels):
    rng = np.random.default_rng(C.derive_seed(net.name, site, "damage_pool"))
    tried, dmgs, within = [], [], []
    block = 100
    while len(tried) < CONFIG["DAMAGE_POOL"] and len(within) < CONFIG["N_DAMAGE"]:
        sets = [np.sort(rng.choice(n_channels, size=k, replace=False)) for _ in range(block)]
        d, _ = damages(net, site, sets)
        for s, dd in zip(sets, d):
            tried.append(s)
            dmgs.append(float(dd))
            if abs(dd - target) <= CONFIG["DAMAGE_TOL"] and len(within) < CONFIG["N_DAMAGE"]:
                within.append(len(tried) - 1)
        print(f"\r    damage-matched search: {len(within)}/{CONFIG['N_DAMAGE']} within "
              f"{CONFIG['DAMAGE_TOL']} after {len(tried)} sets", end="")
    print()
    dmgs = np.array(dmgs)
    if len(within) >= CONFIG["N_DAMAGE"]:
        chosen, how = within[:CONFIG["N_DAMAGE"]], "within tolerance"
    else:
        chosen, how = list(np.argsort(np.abs(dmgs - target))[:CONFIG["N_DAMAGE"]]), "100 closest (tolerance not reached)"
    return [tried[i] for i in chosen], dmgs[chosen], how, len(tried), dmgs


# =============================================================================
# 6. Precision check (TF32 vs full FP32) on a small subset
# =============================================================================
def precision_check(net, site):
    h = HALVES["localizer"]
    sel = np.arange(min(256, len(h.df)))
    y = torch.from_numpy(h.y[sel]).to(DEVICE)

    def run():
        with torch.no_grad():
            a = net.prefix(norm(h.imgs[sel]), site)
            base = p16(net.suffix(a, site)).gather(1, y[:, None])[:, 0]
            drops = []
            for c in range(min(64, a.shape[1])):
                aa = a.clone()
                aa[:, c] = 0
                drops.append((base - p16(net.suffix(aa, site)).gather(1, y[:, None])[:, 0]).cpu().numpy())
        return np.array(drops)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    d_tf32 = run()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d_fp32 = run()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    r = stats.spearmanr(d_tf32.mean(1), d_fp32.mean(1))[0]
    return dict(site=site, channel_mean_spearman=float(r), max_abs_diff=float(np.abs(d_tf32 - d_fp32).max()),
                mean_abs_drop=float(np.abs(d_fp32).mean()))


# =============================================================================
# 7. Run
# =============================================================================
summary_rows, lesion_sets, score_tables = [], {"plan_sha256": PLAN_SHA, "models": {}}, []
for name in CONFIG["MODELS"]:
    print(f"\n[2] {C.MODEL_INFO[name][0]}")
    net = Split(name)
    sites = CONFIG["SITES"][name]
    prec = precision_check(net, sites[min(2, len(sites) - 1)])
    print(f"  precision check at {prec['site']}: TF32 vs FP32 channel-mean Spearman "
          f"{prec['channel_mean_spearman']:0.3f}, max |diff| {prec['max_abs_diff']:0.2e}")
    model_out = dict(precision_check=prec, sites={})
    site_info = {}
    for site in sites:
        print(f"  site {site}")
        sc = site_scores(net, site)
        conf, rule, n_sig = configural_set(sc)
        reliable = bool(sc["sb"] >= CONFIG["REL_THRESHOLD"])
        site_info[site] = dict(sc=sc, conf=conf, rule=rule, n_sig=n_sig, reliable=reliable,
                               sum_I=float(sc["I_loc"][conf].sum()))
        sel = np.where(sc["q_sel"] < CONFIG["Q"])[0]
        jac = len(set(sel) & set(conf)) / max(1, len(set(sel) | set(conf)))
        site_info[site].update(selective=sel, jaccard=float(jac))
        score_tables.append(pd.DataFrame(dict(model=name, site=site, channel=np.arange(sc["n_channels"]),
                                              I_conf_localizer=sc["I_loc"], I_conf_test=sc["I_test"],
                                              local_score=sc["local"], t=sc["t"], p=sc["p"], q=sc["q"],
                                              t_selective=sc["t_sel"], q_selective=sc["q_sel"],
                                              mean_activation=sc["mean_act"])))
        print(f"    reliability: Spearman {sc['spearman']:0.3f}, Spearman-Brown {sc['sb']:0.3f} "
              f"({'reliable' if reliable else 'NOT reliable'}) | configural: {rule} | "
              f"selective: {len(sel)} | Jaccard {jac:0.2f}")

    reliable_sites = [s for s in sites if site_info[s]["reliable"]]
    primary = max(reliable_sites, key=lambda s: site_info[s]["sum_I"]) if reliable_sites else None
    print(f"  primary site: {primary}")
    model_out["primary_site"] = primary

    for site in sites:
        si, sc = site_info[site], site_info[site]["sc"]
        n = sc["n_channels"]
        entry = dict(n_channels=n, reliable=si["reliable"], spearman=sc["spearman"], spearman_brown=sc["sb"],
                     n_fdr_significant=si["n_sig"], configural_rule=si["rule"],
                     configural=si["conf"].tolist(), selective=si["selective"].tolist(),
                     jaccard_selective_configural=si["jaccard"], sum_I_conf=si["sum_I"],
                     mean_activation=sc["mean_act"].tolist())
        if si["reliable"]:
            d_conf, acc0 = damages(net, site, [si["conf"]])
            entry["damage_configural"] = float(d_conf[0])
            entry["baseline_grid_accuracy_localizer"] = float(acc0)
            k = len(si["conf"])
            if site == primary:
                loc_set, loc_dmg = local_set(net, site, sc, d_conf[0])
                entry["local"] = loc_set.tolist()
                entry["damage_local"] = float(loc_dmg)
                rs = random_sets(n, k, CONFIG["N_RANDOM_SIZE"], name, site, "size_matched")
                entry["random_size_matched"] = [s.tolist() for s in rs]
                entry["damage_random_size_matched"] = damages(net, site, rs)[0].tolist()
                dm, dmd, how, n_tried, _ = damage_matched(net, site, k, d_conf[0], n)
                entry["random_damage_matched"] = [s.tolist() for s in dm]
                entry["damage_random_damage_matched"] = dmd.tolist()
                entry["damage_matching"] = dict(rule=how, sets_tried=n_tried)
                sel_sets = [si["selective"]] if len(si["selective"]) else []
                if sel_sets:
                    entry["damage_selective"] = float(damages(net, site, sel_sets)[0][0])
                dose = {}
                order = np.argsort(-sc["I_loc"])
                for f in CONFIG["DOSE_FRACS"]:
                    kk = max(1, int(round(f * n)))
                    dose[str(f)] = dict(configural=np.sort(order[:kk]).tolist(),
                                        random=[s.tolist() for s in random_sets(n, kk, CONFIG["DOSE_RANDOM"], name, site, "dose", f)])
                entry["dose_response"] = dose
            else:
                entry["random_size_matched"] = [s.tolist() for s in
                                                random_sets(n, k, CONFIG["OTHER_SITE_RANDOM"], name, site, "other_site")]
        model_out["sites"][site] = entry
        summary_rows.append(dict(model=name, site=site, channels=n, spearman_brown=sc["sb"], reliable=si["reliable"],
                                 n_fdr=si["n_sig"], configural_size=len(si["conf"]),
                                 configural_pct=100 * len(si["conf"]) / n, sum_I=si["sum_I"],
                                 selective_size=len(si["selective"]), jaccard=si["jaccard"],
                                 damage_configural=entry.get("damage_configural", np.nan),
                                 local_size=len(entry.get("local", [])), damage_local=entry.get("damage_local", np.nan),
                                 primary=(site == primary)))

    # exploratory pilot on the calibration test half
    if primary is not None:
        e = model_out["sites"][primary]
        h = HALVES["test"]
        pilot_random = random_sets(e["n_channels"], len(e["configural"]), CONFIG["PILOT_RANDOM"], name, primary, "pilot")
        sets = [e["configural"], e["local"]] + [s.tolist() for s in pilot_random]
        rows = eval_sets(net, primary, h, sets)
        crs = [cr(r, h) for r in rows]
        base, conf_, loc_ = crs[0], crs[1], crs[2]
        rand = np.array([c[0] for c in crs[3:]], dtype=float)
        fmt = lambda v: "floor" if not np.isfinite(v) else f"{v:0.3f}"
        pilot = dict(baseline_CR=base[0], baseline_grid=base[1], baseline_scr=base[2],
                     configural_CR=conf_[0], configural_grid=conf_[1], configural_scr=conf_[2],
                     local_CR=loc_[0], local_grid=loc_[1], local_scr=loc_[2],
                     random_CR_mean=float(np.nanmean(rand)) if np.isfinite(rand).any() else None,
                     random_CR_min=float(np.nanmin(rand)) if np.isfinite(rand).any() else None,
                     random_CR_max=float(np.nanmax(rand)) if np.isfinite(rand).any() else None,
                     random_at_floor=int((~np.isfinite(rand)).sum()),
                     note="exploratory, calibration test half, zero ablation, size-matched random sets")
        model_out["pilot_test_half"] = pilot
        rtxt = (f"{np.nanmean(rand):0.3f} [{np.nanmin(rand):0.3f}, {np.nanmax(rand):0.3f}]"
                if np.isfinite(rand).any() else "all at floor")
        print(f"  pilot (test half, exploratory): CR intact {fmt(base[0])} | configural lesion {fmt(conf_[0])} "
              f"(grid {conf_[1]:0.3f}, scr {conf_[2]:0.3f}) | local lesion {fmt(loc_[0])} | random {rtxt}"
              f"{f' ({int((~np.isfinite(rand)).sum())} at floor)' if (~np.isfinite(rand)).any() else ''}")
    lesion_sets["models"][name] = model_out
    del net
    torch.cuda.empty_cache()

# =============================================================================
# 8. Save, table, figure
# =============================================================================
out_json = RES / f"lesion_sets{'_smoke' if SMOKE else ''}.json"
C.save_json(lesion_sets, out_json)
sets_sha = hashlib.sha256(out_json.read_bytes()).hexdigest()
pd.concat(score_tables).to_csv(RES / f"channel_scores{'_smoke' if SMOKE else ''}.csv", index=False)
summ = pd.DataFrame(summary_rows)
summ.to_csv(RES / f"stage3_summary{'_smoke' if SMOKE else ''}.csv", index=False)

lines = [r"\begin{tabular}{llrrrrrrl}", r"\hline",
         r"Model & Site & Channels & Reliability & FDR-significant & Configural set & Selective & Jaccard & Damage \\",
         r"\hline"]
for r in summ.itertuples():
    lines.append(f"{C.MODEL_INFO[r.model][0]} & {r.site}{' (primary)' if r.primary else ''} & {r.channels} & "
                 f"{r.spearman_brown:0.2f} & {r.n_fdr} & {r.configural_size} ({r.configural_pct:0.1f}\\%) & "
                 f"{r.selective_size} & {r.jaccard:0.2f} & "
                 f"{'--' if np.isnan(r.damage_configural) else f'{r.damage_configural:0.3f}'} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage3_table.tex").write_text("\n".join(lines))


def fig_reliability():
    """Split-half reliability of configural importance at each model's primary site."""
    C.set_pub_style()
    models = [m for m in CONFIG["MODELS"] if lesion_sets["models"][m]["primary_site"]]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(8.5 * len(models), 7.5), squeeze=False)
    for p, (ax, m) in enumerate(zip(axes[0], models)):
        site = lesion_sets["models"][m]["primary_site"]
        d = pd.concat(score_tables)
        d = d[(d.model == m) & (d.site == site)]
        conf = set(lesion_sets["models"][m]["sites"][site]["configural"])
        is_c = d.channel.isin(conf).to_numpy()
        x, y = d.I_conf_localizer.to_numpy(), d.I_conf_test.to_numpy()
        lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) or 1.0
        e = int(np.floor(np.log10(lim)))
        x, y, lim = x / 10.0 ** e, y / 10.0 ** e, lim / 10.0 ** e      # lim now in [1, 10)
        ax.scatter(x[~is_c], y[~is_c], s=14, color="#9a9a9a", alpha=0.6, lw=0, label="Other channels")
        ax.scatter(x[is_c], y[is_c], s=22, color="#2a78d6", alpha=0.9, lw=0, label="Configural set")
        step = 0.5 if lim <= 2.5 else (1.0 if lim <= 5 else 2.0)
        bound = np.ceil(lim / step - 1e-9) * step
        ax.plot([-bound, bound], [-bound, bound], color="0.3", lw=1.0, ls="--")
        dec = 1 if step < 1 else 0
        C.style_axes(ax, xlim=(-bound, bound), xstep=step, ylim=(-bound, bound), ystep=step, decimals=dec)
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%0.{dec}f"))
        unit = "" if e == 0 else f" ($\\times 10^{{{e}}}$)"
        ax.set_xlabel(f"Configural importance, localizer half{unit}")
        ax.set_ylabel(f"Configural importance, test half{unit}")
        ax.yaxis.set_label_coords(-0.14, 0.5)
        ax.text(-0.14, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
        e = lesion_sets["models"][m]["sites"][site]
        ax.text(0.04, 0.96, f"{C.MODEL_INFO[m][0]}, {site}\nSpearman-Brown {e['spearman_brown']:0.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=14)
        leg = ax.legend(loc="lower right")
        C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.35)
    C.save_figure(fig, FIGS / f"stage3_reliability{'_smoke' if SMOKE else ''}")
    plt.close(fig)


fig_reliability()

pd.set_option("display.width", 220)
print("\n" + summ.round(3).to_string(index=False))
print("\n" + "=" * 78)
for m in CONFIG["MODELS"]:
    mo = lesion_sets["models"][m]
    ps = mo["primary_site"]
    if ps is None:
        print(f"  {C.MODEL_INFO[m][0]}: NO RELIABLE SITE (no lesion claim for this model, per plan)")
        continue
    e = mo["sites"][ps]
    print(f"  {C.MODEL_INFO[m][0]}: primary site {ps} | configural {len(e['configural'])} channels "
          f"({e['configural_rule']}) | damage {e['damage_configural']:0.3f} | local {len(e['local'])} channels, "
          f"damage {e['damage_local']:0.3f} | damage-matched random: {e['damage_matching']['rule']}")
print(f"  lesion_sets.json SHA-256: {sets_sha}")
print("=" * 78)
print("\nDone. Outputs in", RES, "and", FIGS)
