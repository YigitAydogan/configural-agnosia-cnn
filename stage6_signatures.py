"""
STAGE 6: Agnosia signature tests on the hold-out set (H4 to H7, secondary family)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 1 to 1.5 hours on an A100). Results
are saved per model, ablation method and lesion condition; if the runtime disconnects, rerunning picks
up where it stopped.

Same checks as Stage 5 before any hold-out image is used (frozen plan, Amendment A1, the lesion sets
published in v1.2, hold-out fingerprint); every hold-out run is logged.

Conditions (lesion sets from lesion_sets_A1.json, primary site): intact, configural, local (descriptive)
and the 100 damage-matched random sets; zero ablation (primary) and mean ablation (robustness).

Features and probes (plan: Agnosia signature tests)
  Penultimate features (ResNet-50: global pool, 2048; CORnet-S: decoder pool, 512). Linear L2 logistic
  regression, retrained for every condition on calibration features, regularisation chosen by 5-fold
  cross-validation grouped by object (grid 1e-4 to 1), then tested on hold-out features.

  H4  Local evidence kept. Probes on scrambled images: dominant hue (6 bins of 60 degrees over object
      pixels with saturation > 0.2) and category. Drop = intact minus lesioned probe accuracy.
      Supported if, for BOTH probes, drop(configural) - mean drop(damage-matched random) has upper 95%
      bound < 0.05. The same category probe on grid controls is reported (whole-object loss).
  H5  Errors become local. On hold-out grid controls, the off-diagonal confusion distribution
      (category-balanced error rates, normalised to sum to 1) is compared with BagNet-33's by
      Jensen-Shannon distance (base 2). Supported if [JS(configural) - JS(intact)] minus the mean of
      the same change over random lesions has upper bound < 0.
  H6  Line-drawing advantage lost. Edge minus silhouette accuracy over the same 160 public images
      (paired by image). Supported if its change after the configural lesion minus the mean change
      after random lesions has upper bound < 0.
  H7  Overlapping figures. A 16-output multi-label probe trained on calibration outlines and pairs
      (single, side by side, small and large overlap). A hold-out pair is correct when the probe's top
      two outputs are its two categories. Overlap cost = side-by-side minus small-overlap rate.
      Supported if its change after the configural lesion minus the mean change after random lesions
      has lower bound > 0.

  Decision: rule met AND Holm-adjusted two-sided bootstrap p < 0.05 within the secondary family
  (H4 to H7 x 2 models = 8 tests). H4's p is the larger of its two probes' p-values against the 0.05
  margin (intersection-union). Robust only if zero and mean ablation agree (as in Stage 5).

Implementation details the plan leaves open, fixed here before any result is seen
  - Hue labels come from each object's grid-control image; objects with fewer than 100 saturated
    pixels have no dominant hue and are left out of the hue probe.
  - Control tasks (labels shuffled at object level, Hewitt & Liang): hold-out objects are new, so a
    shuffled-label probe is at chance there by construction. Selectivity is therefore measured on the
    calibration set by holding out one scramble variant per object (task and control alike), for the
    two scrambled-image probes, in the intact, configural and local conditions. Grid and multi-label
    probes have one image per object, so no control task is defined for them.
  - Bootstrap: category-stratified over hold-out objects (H4, H5), over the 160 image pairs stratified
    by category (H6), over hold-out pairs stratified by the first object's category (H7); probes are
    fitted once per condition, so intervals cover test-set sampling.

Outputs: results/stage6/*.csv|json|tex, figures/stage6/*.pdf|png
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
from matplotlib.colors import rgb_to_hsv
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
sfx = "_smoke" if SMOKE else ""

PLAN_SHA_REGISTERED = "dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c"
AMENDMENT_SHA = "6dd4a7445862ba90f392a6028623abdf4d98340ada89fa8ff6f396bc5b814e9f"
SETS_SHA_REGISTERED = "9bb05eb1ee8729786246ff80f23575225c88f3271c4c6756a92525500c2fb904"

assert C.FROZEN_PLAN.exists(), "The analysis plan is not frozen."
FROZEN = C.load_json(C.FROZEN_PLAN)
PLAN = FROZEN["plan"]
ok, why = C.holdout_status()
assert ok, f"Hold-out locked: {why}"
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{sfx}.json"
sets_sha = hashlib.sha256(SETS_PATH.read_bytes()).hexdigest()
if not SMOKE:
    assert FROZEN["plan_sha256"] == PLAN_SHA_REGISTERED, "Frozen plan differs from the registered plan."
    assert hashlib.sha256((C.PLAN_DIR / "amendment_01.json").read_bytes()).hexdigest() == AMENDMENT_SHA
    assert sets_sha == SETS_SHA_REGISTERED, "lesion_sets_A1.json differs from the version published in v1.2."
SETS = C.load_json(SETS_PATH)

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    REFERENCE=PLAN["models"]["local_reference"],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    MARGIN=PLAN["statistics"]["margin"],
    N_HUE_BINS=6, HUE_MIN_SAT=0.2, HUE_MIN_PIXELS=100,
    LAMBDAS=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    N_FOLDS=5, LBFGS_ITERS=150,
    N_BOOT=10000, ALPHA=0.05,
    BATCH=128, CHUNK=8,
)
if SMOKE:
    CONFIG.update(N_BOOT=300, BATCH=32, MIN_OBJECTS_PER_CATEGORY=2, LBFGS_ITERS=40, CHUNK=4)

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True     # as in Stages 3 and 5
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage6"
CKPT = RES / "checkpoints"
FIGS = C.FIG_DIR / "stage6"
for d in [RES, CKPT, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
print(f"Stage 6 (secondary hypotheses) | plan {FROZEN['plan_sha256'][:12]} | lesion sets {sets_sha[:12]} | device {DEVICE}")

# =============================================================================
# 1. Images
# =============================================================================
if IN_COLAB:
    for st in ["frozen_calib", "stage1", C.HOLDOUT_STAGE]:
        C.unpack_stimuli(st)
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
spec = C.load_json(C.HOLDOUT_SPEC)
t0 = time.time()
assert C.fingerprint_files(C.STIM_LOCAL, hm.path.tolist()) == spec["fingerprint"], "Hold-out files changed since locking."
print(f"  hold-out fingerprint verified ({len(hm)} files, {time.time() - t0:0.0f} s)")
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
s1 = pd.read_csv(C.STIM_DRIVE / "stage1_manifest.csv", low_memory=False)


def configural_table(man, exclude=()):
    d = man[(man.battery == "configural_tex") & man.condition.isin(["cutG_intact", "scrambled"])].copy()
    d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_STRENGTH"]))]
    d = d[~d.category.isin(exclude)]
    keep = set(d[d.condition == "cutG_intact"].source_id) & set(d[d.condition == "scrambled"].source_id)
    return d[d.source_id.isin(keep)].sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)


cal = configural_table(fc)
ho = configural_table(hm, CONFIG["EXCLUDED"])
n_per_cat = ho.drop_duplicates("source_id").category.value_counts()
small = sorted(n_per_cat[n_per_cat < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)
ho = ho[~ho.category.isin(small)].reset_index(drop=True)
if SMOKE:
    cal = cal[cal.source_id.isin(sorted(cal.source_id.unique())[:48])].reset_index(drop=True)
ovc = s1[s1.battery == "overlap"].reset_index(drop=True)
ovh = hm[hm.battery == "overlap"].reset_index(drop=True)
ref = s1[s1.condition.isin(["ref_edge", "ref_silhouette"])].reset_index(drop=True)
assert not cal.path.map(C.is_holdout_path).any() and not ovc.path.map(C.is_holdout_path).any()

PARTS = [("cal", cal), ("ho", ho), ("ovc", ovc), ("ovh", ovh), ("ref", ref)]
ALL = pd.concat([d.assign(part=k) for k, d in PARTS], ignore_index=True)
OFF = {}
o = 0
for k, d in PARTS:
    OFF[k] = (o, o + len(d))
    o += len(d)
size = 64 if SMOKE else C.IMG_SIZE
t0 = time.time()
IMGS = torch.from_numpy(np.stack([np.array(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)), dtype=np.uint8)
                                  for p in ALL.path])).permute(0, 3, 1, 2).contiguous()
HO_PATHS = [p for p in ALL.path if C.is_holdout_path(p)]
print(f"  images: calibration {len(cal)}, hold-out {len(ho)}, overlap {len(ovc)} + {len(ovh)}, reference {len(ref)} "
      f"| loaded in {time.time() - t0:0.0f} s | excluded hold-out categories: {', '.join(CONFIG['EXCLUDED'] + small)}")


class Objects:
    """Object bookkeeping for a configural table: grid image per object, scramble averaging, labels."""

    def __init__(self, d, part):
        self.d, self.part = d, part
        self.objects = sorted(d.source_id.unique())
        oi = {o: i for i, o in enumerate(self.objects)}
        self.obj = d.source_id.map(oi).to_numpy()
        self.is_grid = (d.condition == "cutG_intact").to_numpy()
        self.y_img = d.category.map(C.CAT_TO_IDX).to_numpy()
        self.cat = d.drop_duplicates("source_id").set_index("source_id").loc[self.objects, "category"].to_numpy()
        self.y_obj = np.array([C.CAT_TO_IDX[c] for c in self.cat])
        self.grid = np.array([np.where(self.is_grid & (self.obj == i))[0][0] for i in range(len(self.objects))])
        self.scr = np.where(~self.is_grid)[0]
        S = np.zeros((len(self.scr), len(self.objects)), np.float32)
        for j, i in enumerate(self.obj[self.scr]):
            S[j, i] = 1.0
        self.scr_avg = S / S.sum(0, keepdims=True)
        self.variant = d.variant.to_numpy()
        # dominant hue from the grid-control image (object pixels with saturation > 0.2)
        a0 = OFF[part][0]
        hue = np.full(len(self.objects), -1)
        for i, g in enumerate(self.grid):
            img = IMGS[a0 + g].permute(1, 2, 0).numpy().astype(np.float32) / 255.0
            hsv = rgb_to_hsv(img)
            m = hsv[..., 1] > CONFIG["HUE_MIN_SAT"]
            if m.sum() >= (CONFIG["HUE_MIN_PIXELS"] * (size / C.IMG_SIZE) ** 2):
                h = np.floor(hsv[..., 0][m] * CONFIG["N_HUE_BINS"]).astype(int) % CONFIG["N_HUE_BINS"]
                hue[i] = np.bincount(h, minlength=CONFIG["N_HUE_BINS"]).argmax()
        self.hue_obj = hue


CAL, HO = Objects(cal, "cal"), Objects(ho, "ho")
print(f"  hue labels: calibration {(CAL.hue_obj >= 0).sum()} of {len(CAL.objects)} objects, "
      f"hold-out {(HO.hue_obj >= 0).sum()} of {len(HO.objects)}; hold-out hue bins "
      f"{np.bincount(HO.hue_obj[HO.hue_obj >= 0], minlength=CONFIG['N_HUE_BINS']).tolist()}")


def multihot(d):
    Y = np.zeros((len(d), C.N_CAT), np.float32)
    for i, r in enumerate(d.itertuples()):
        Y[i, C.CAT_TO_IDX[r.category]] = 1
        if isinstance(getattr(r, "category_b", None), str):
            Y[i, C.CAT_TO_IDX[r.category_b]] = 1
    return Y


Y_OVC, Y_OVH = multihot(ovc), multihot(ovh)
G_OVC = ovc.source_id.astype(str).to_numpy()
pairs = ovh[ovh.condition.isin(["side_by_side", "overlap_small", "overlap_large"])]
PAIR_IDS = sorted(pairs.source_id.unique())
PAIR_ROWS = {c: np.array([pairs[(pairs.source_id == p) & (pairs.condition == c)].index[0] for p in PAIR_IDS])
             for c in ["side_by_side", "overlap_small", "overlap_large"]}
PAIR_CAT = pairs.drop_duplicates("source_id").set_index("source_id").loc[PAIR_IDS, "category"].to_numpy()
stem = lambda p: f"{Path(p).parent.name}/{Path(p).stem}"
ref_e = ref[ref.condition == "ref_edge"].reset_index()
ref_s = ref[ref.condition == "ref_silhouette"].reset_index()
REF_KEYS = sorted(set(ref_e.path.map(stem)) & set(ref_s.path.map(stem)))
REF_E = np.array([ref_e.index[ref_e.path.map(stem) == k][0] for k in REF_KEYS])
REF_S = np.array([ref_s.index[ref_s.path.map(stem) == k][0] for k in REF_KEYS])
REF_CAT = np.array([k.split("/")[0] for k in REF_KEYS])
REF_Y = np.array([C.CAT_TO_IDX[c] for c in REF_CAT])
REF_E_ROWS, REF_S_ROWS = ref_e.loc[REF_E, "index"].to_numpy(), ref_s.loc[REF_S, "index"].to_numpy()
print(f"  overlap: {len(PAIR_IDS)} hold-out pairs | reference: {len(REF_KEYS)} edge/silhouette image pairs")

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def norm(x):
    return (x.to(DEVICE, non_blocking=True).float() / 255.0 - MEAN) / STD


# =============================================================================
# 2. Models: prefix to the site, suffix to (penultimate features, logits)
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
        elif name == "cornet_s":
            self.stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.feat = lambda x: m.decoder.flatten(m.decoder.avgpool(x))
            self.fc = m.decoder.linear
        else:
            self.stages = [("stem", nn.Sequential(m.conv1, m.conv2, m.bn1, m.relu)), ("layer1", m.layer1),
                           ("layer2", m.layer2), ("layer3", m.layer3), ("layer4", m.layer4)]
            self.feat = lambda x: x.mean(dim=(2, 3))
            self.fc = m.fc
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


@torch.no_grad()
def run_conditions(net, site, conds, mean_act):
    """conds: list of (channels, mode). Returns per condition (features on GPU, 16-way decisions)."""
    n = len(ALL)
    feats = [None] * len(conds)
    decs = [np.zeros(n, np.int64) for _ in conds]
    mv = None if mean_act is None else torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
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
                feats[k] = torch.zeros(n, f.shape[1], device=DEVICE)
            feats[k][b0:b0 + len(f)] = f
            decs[k][b0:b0 + len(f)] = p.cpu().numpy()
    return feats, decs


# =============================================================================
# 3. Probes: batched L2 logistic regression (L-BFGS), lambda by grouped 5-fold CV
# =============================================================================
def folds_for(groups):
    ug = np.unique(groups)
    rng = np.random.default_rng(C.derive_seed("stage6_folds", len(ug)))
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


def selectivity(Xtr_all, y_obj_all, obj_all, variant_all, lam, K, seed_key):
    """Calibration-only control task: hold out each object's last scramble variant; task labels vs labels
    shuffled across objects. Returns task accuracy, control accuracy, selectivity."""
    last = pd.Series(variant_all).groupby(obj_all).transform("max").to_numpy()
    te = (variant_all == last) & (pd.Series(obj_all).map(pd.Series(obj_all).value_counts()).to_numpy() > 1)
    tr = ~te
    rng = np.random.default_rng(C.derive_seed("stage6_control", seed_key))
    perm = rng.permutation(len(y_obj_all))
    res = []
    for labels_obj in [y_obj_all, y_obj_all[perm]]:
        y = labels_obj[obj_all]
        Xa, Xb = standardise(Xtr_all[torch.as_tensor(tr, device=DEVICE)], Xtr_all[torch.as_tensor(te, device=DEVICE)])
        W, b = fit(Xa, y[tr], np.ones((1, tr.sum()), np.float32), [lam], K)
        with torch.no_grad():
            pred = (Xb @ W[0] + b[0]).argmax(1).cpu().numpy()
        res.append(float((pred == y[te]).mean()))
    return res[0], res[1], res[0] - res[1]


def condition_outputs(feat, dec, want_selectivity, key):
    """All probe predictions and model decisions needed for H4 to H7 under one condition."""
    out = {}
    c0, c1 = OFF["cal"]
    h0, h1 = OFF["ho"]
    Fc, Fh = feat[c0:c1], feat[h0:h1]
    # H4: scrambled-image probes (hue, category); grid-control category probe (reported)
    cs, hs = torch.as_tensor(CAL.scr, device=DEVICE), torch.as_tensor(HO.scr, device=DEVICE)
    hue_ok = CAL.hue_obj[CAL.obj[CAL.scr]] >= 0
    tr = cs[torch.as_tensor(hue_ok, device=DEVICE)]
    sc, lam_h = probe(Fc[tr], CAL.hue_obj[CAL.obj[CAL.scr]][hue_ok], CAL.obj[CAL.scr][hue_ok], Fh[hs], CONFIG["N_HUE_BINS"])
    out["hue_pred"] = sc.argmax(1)
    sc, lam_c = probe(Fc[cs], CAL.y_img[CAL.scr], CAL.obj[CAL.scr], Fh[hs], C.N_CAT)
    out["catscr_pred"] = sc.argmax(1)
    cg, hg = torch.as_tensor(CAL.grid, device=DEVICE), torch.as_tensor(HO.grid, device=DEVICE)
    sc, _ = probe(Fc[cg], CAL.y_obj, np.arange(len(CAL.objects)), Fh[hg], C.N_CAT)
    out["catgrid_pred"] = sc.argmax(1)
    if want_selectivity:
        Xs = Fc[cs]
        out["sel_hue"] = selectivity(Xs[torch.as_tensor(hue_ok, device=DEVICE)],
                                     np.where(CAL.hue_obj >= 0, CAL.hue_obj, 0), CAL.obj[CAL.scr][hue_ok],
                                     CAL.variant[CAL.scr][hue_ok], lam_h, CONFIG["N_HUE_BINS"], key + "hue")
        out["sel_cat"] = selectivity(Xs, CAL.y_obj, CAL.obj[CAL.scr], CAL.variant[CAL.scr], lam_c, C.N_CAT, key + "cat")
    # H5: decisions on hold-out grid controls; H6: decisions on reference images
    out["dec_grid"] = dec[h0:h1][HO.grid]
    r0, _ = OFF["ref"]
    out["dec_edge"] = dec[r0 + REF_E_ROWS]
    out["dec_sil"] = dec[r0 + REF_S_ROWS]
    # H7: multi-label probe, top-2 on hold-out pairs
    o0, o1 = OFF["ovc"]
    v0, v1 = OFF["ovh"]
    sc, _ = probe(feat[o0:o1], Y_OVC, G_OVC, feat[v0:v1], C.N_CAT, multilabel=True)
    for c, rows in PAIR_ROWS.items():
        top2 = np.argsort(-sc[rows], 1)[:, :2]
        truth = Y_OVH[rows]
        out[f"pair_{c}"] = np.array([truth[i, top2[i]].sum() == 2 for i in range(len(rows))])
    return out


# =============================================================================
# 4. Run (checkpointed per model, ablation method and condition)
# =============================================================================
def conditions_for(name):
    e = SETS["models"][name]["sites"][SETS["models"][name]["primary_site"]]
    base = [("intact", [], "zero"), ("configural", e["configural"], "zero"), ("local", e["local"], "zero")]
    base += [(f"damage {i + 1}", s, "zero") for i, s in enumerate(e["random_damage_matched"])]
    mean = [(f"{l} (mean)", s, "mean") for l, s, _ in base[1:]]
    return base + mean


SEL_CONDS = {"intact", "configural", "local", "configural (mean)", "local (mean)"}
OUT = {}
for name in CONFIG["MODELS"] + [CONFIG["REFERENCE"]]:
    print(f"\n[{C.MODEL_INFO[name][0]}]")
    is_ref = name == CONFIG["REFERENCE"]
    conds = [("intact", [], "zero")] if is_ref else conditions_for(name)
    site = "layer4" if is_ref else SETS["models"][name]["primary_site"]
    mean_act = None if is_ref else SETS["models"][name]["sites"][site]["mean_activation"]
    OUT[name] = {}
    todo = []
    for lab, ch, mode in conds:
        ck = CKPT / f"{name}_{lab.replace(' ', '_').replace('(', '').replace(')', '')}{sfx}.npz"
        if ck.exists():
            z = np.load(ck, allow_pickle=True)
            OUT[name][lab] = {k: z[k] for k in z.files}
        else:
            todo.append((lab, ch, mode, ck))
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
        for (lab, ch, mode, ck), f, dd in zip(chunk, feats, decs):
            if is_ref:
                h0, h1 = OFF["ho"]
                o = dict(dec_grid=dd[h0:h1][HO.grid])
            else:
                o = condition_outputs(f, dd, lab in SEL_CONDS, f"{name}_{lab}")
            np.savez_compressed(ck, **{k: np.asarray(v) for k, v in o.items()})
            OUT[name][lab] = {k: np.asarray(v) for k, v in o.items()}
        del feats
        torch.cuda.empty_cache()
        done = c0 + len(chunk)
        el = time.time() - t0
        print(f"    {done}/{len(todo)} conditions | {el / 60:0.1f} min | ETA {el / done * (len(todo) - done) / 60:0.1f} min")
    del net
    torch.cuda.empty_cache()

# =============================================================================
# 5. Statistics
# =============================================================================
def strat_weights(cats, key):
    """Point weights and (N_BOOT, n) bootstrap weights: category-balanced, resampled within category."""
    rng = np.random.default_rng(C.derive_seed("stage6_boot", key))
    u = sorted(set(cats))
    n = len(cats)
    W = np.zeros((CONFIG["N_BOOT"], n), np.float32)
    w0 = np.zeros(n, np.float32)
    for c in u:
        g = np.where(cats == c)[0]
        s = 1.0 / (len(g) * len(u))
        w0[g] = s
        draws = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
        cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
        np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), draws.ravel()), 1)
        W[:, g] = cnt * s
    return w0, W


WO = strat_weights(HO.cat, "objects")
hue_obj_ok = HO.hue_obj >= 0
WH = strat_weights(HO.cat[hue_obj_ok], "hue_objects")
WR = strat_weights(REF_CAT, "reference")
WP = strat_weights(PAIR_CAT, "pairs")


def wmean(w, x):
    """Point (scalar) and bootstrap (N_BOOT,) weighted means of per-unit values x (n,) or (n, k)."""
    w0, W = w
    return w0 @ x, W @ x


def per_object_scr(correct_scr):
    return correct_scr.astype(np.float32) @ HO.scr_avg


def summarize(point, boot, ref_value=0.0):
    boot = np.asarray(boot, float)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    n = len(boot)
    p = min(1.0, 2 * min((1 + (boot <= ref_value).sum()) / (n + 1), (1 + (boot >= ref_value).sum()) / (n + 1)))
    return dict(estimate=float(point), lo=float(lo), hi=float(hi), p_two_sided=float(p))


def measures(o):
    """Per-unit arrays for every measure under one condition."""
    m = {}
    scr_y = HO.y_img[HO.scr]
    m["hue"] = per_object_scr(o["hue_pred"] == HO.hue_obj[HO.obj[HO.scr]])[hue_obj_ok]
    m["catscr"] = per_object_scr(o["catscr_pred"] == scr_y)
    m["catgrid"] = (o["catgrid_pred"] == HO.y_obj).astype(np.float32)
    m["edge_minus_sil"] = (o["dec_edge"] == REF_Y).astype(np.float32) - (o["dec_sil"] == REF_Y).astype(np.float32)
    m["overlap_cost"] = o["pair_side_by_side"].astype(np.float32) - o["pair_overlap_small"].astype(np.float32)
    m["pair_sbs"] = o["pair_side_by_side"].astype(np.float32)
    m["pair_small"] = o["pair_overlap_small"].astype(np.float32)
    m["pair_large"] = o["pair_overlap_large"].astype(np.float32)
    return m


def confusion_onehot(dec):
    E = np.zeros((len(dec), C.N_CAT * C.N_CAT), np.float32)
    E[np.arange(len(dec)), HO.y_obj * C.N_CAT + dec] = 1
    return E


OFFDIAG = ~np.eye(C.N_CAT, dtype=bool).ravel()


def js_distance(Pm, Qm):
    """Rows: (B, 256) category-balanced (true, predicted) rates. JS distance (base 2) between the
    off-diagonal error distributions."""
    P, Q = Pm[:, OFFDIAG], Qm[:, OFFDIAG]
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
    M_ = 0.5 * (P + Q)
    kl = lambda a, b: np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0).sum(1)
    return np.sqrt(np.clip(0.5 * kl(P, M_) + 0.5 * kl(Q, M_), 0, None))


ref_E = confusion_onehot(OUT[CONFIG["REFERENCE"]]["intact"]["dec_grid"])
ref_p, ref_b = wmean(WO, ref_E)
ref_p = ref_p[None]


def js_to_ref(dec):
    p, b = wmean(WO, confusion_onehot(dec))
    return js_distance(p[None], ref_p)[0], js_distance(b, ref_b)


rows, desc_rows, sel_rows = [], [], []
for name in CONFIG["MODELS"]:
    for mode, tag in [("zero", ""), ("mean", " (mean)")]:
        base = OUT[name]["intact"]
        conf = OUT[name][f"configural{tag}"]
        n_dm = len(SETS["models"][name]["sites"][SETS["models"][name]["primary_site"]]["random_damage_matched"])
        rand = [OUT[name][f"damage {i + 1}{tag}"] for i in range(n_dm)]
        mb, mc = measures(base), measures(conf)
        mr = [measures(r) for r in rand]
        res = {}
        # H4: drops (intact - lesion), configural minus mean random, non-inferiority margin
        for pr, w in [("hue", WH), ("catscr", WO)]:
            bp, bb = wmean(w, mb[pr])
            cp, cb = wmean(w, mc[pr])
            R = np.stack([x[pr] for x in mr], 1)
            rp, rb = wmean(w, R)
            d_p = (bp - cp) - np.mean(bp - rp)
            d_b = (bb - cb) - np.mean(bb[:, None] - rb, 1)
            res[f"H4 {pr}"] = summarize(d_p, d_b, ref_value=CONFIG["MARGIN"])
            desc_rows.append(dict(model=name, ablation=mode, measure=f"probe accuracy: {pr}", intact=float(bp),
                                  configural=float(cp), random_mean=float(np.mean(rp)),
                                  random_lo=float(np.quantile(rp, 0.025)), random_hi=float(np.quantile(rp, 0.975))))
        bp, _ = wmean(WO, mb["catgrid"])
        cp, _ = wmean(WO, mc["catgrid"])
        rp = np.array([wmean(WO, x["catgrid"])[0] for x in mr])
        desc_rows.append(dict(model=name, ablation=mode, measure="probe accuracy: catgrid", intact=float(bp),
                              configural=float(cp), random_mean=float(rp.mean()),
                              random_lo=float(np.quantile(rp, 0.025)), random_hi=float(np.quantile(rp, 0.975))))
        # H5: JS distance to BagNet-33 errors
        jb_p, jb_b = js_to_ref(base["dec_grid"])
        jc_p, jc_b = js_to_ref(conf["dec_grid"])
        jr = [js_to_ref(r["dec_grid"]) for r in rand]
        jr_p = np.array([x[0] for x in jr])
        jr_b = np.stack([x[1] for x in jr], 1)
        res["H5"] = summarize((jc_p - jb_p) - np.mean(jr_p - jb_p), (jc_b - jb_b) - np.mean(jr_b - jb_b[:, None], 1))
        desc_rows.append(dict(model=name, ablation=mode, measure="JS distance to BagNet-33 errors", intact=float(jb_p),
                              configural=float(jc_p), random_mean=float(jr_p.mean()),
                              random_lo=float(np.quantile(jr_p, 0.025)), random_hi=float(np.quantile(jr_p, 0.975))))
        # H6 and H7: change after lesion, configural minus mean random
        for h, meas, w in [("H6", "edge_minus_sil", WR), ("H7", "overlap_cost", WP)]:
            bp, bb = wmean(w, mb[meas])
            cp, cb = wmean(w, mc[meas])
            R = np.stack([x[meas] for x in mr], 1)
            rp, rb = wmean(w, R)
            res[h] = summarize((cp - bp) - np.mean(rp - bp), (cb - bb) - np.mean(rb - bb[:, None], 1))
            desc_rows.append(dict(model=name, ablation=mode, measure=meas, intact=float(bp), configural=float(cp),
                                  random_mean=float(np.mean(rp)), random_lo=float(np.quantile(rp, 0.025)),
                                  random_hi=float(np.quantile(rp, 0.975))))
        for meas in ["pair_sbs", "pair_small", "pair_large"]:
            bp, _ = wmean(WP, mb[meas])
            cp, _ = wmean(WP, mc[meas])
            rp = np.array([wmean(WP, x[meas])[0] for x in mr])
            desc_rows.append(dict(model=name, ablation=mode, measure=meas, intact=float(bp), configural=float(cp),
                                  random_mean=float(rp.mean()), random_lo=float(np.quantile(rp, 0.025)),
                                  random_hi=float(np.quantile(rp, 0.975))))
        for k, v in res.items():
            hyp = k.split()[0]
            rule = dict(H4=v["hi"] < CONFIG["MARGIN"], H5=v["hi"] < 0, H6=v["hi"] < 0, H7=v["lo"] > 0)[hyp]
            rows.append(dict(model=name, ablation=mode, hypothesis=hyp, contrast=k, n_random=len(rand), **v,
                             rule_met=bool(rule)))
        for lab in (["intact"] if mode == "zero" else []) + [f"configural{tag}", f"local{tag}"]:
            o = OUT[name][lab]
            if "sel_hue" in o:
                for pr in ["hue", "cat"]:
                    t, c, s = o[f"sel_{pr}"]
                    sel_rows.append(dict(model=name, condition=lab, probe=pr, task=float(t), control=float(c),
                                         selectivity=float(s)))

hyp = pd.DataFrame(rows)
# decision per hypothesis (zero ablation), Holm over 8 tests
fam = []
for (m, h), g in hyp[hyp.ablation == "zero"].groupby(["model", "hypothesis"], sort=False):
    fam.append(dict(model=m, hypothesis=h, p=float(g.p_two_sided.max()), rule_met=bool(g.rule_met.all())))
fam = pd.DataFrame(fam).sort_values("p").reset_index(drop=True)
adj, run = [], 0.0
for i, p in enumerate(fam.p):
    run = max(run, min(1.0, (len(fam) - i) * p))
    adj.append(run)
fam["p_holm"] = adj
fam["supported"] = fam.rule_met & (fam.p_holm < CONFIG["ALPHA"])
mean_rule = hyp[hyp.ablation == "mean"].groupby(["model", "hypothesis"]).rule_met.all()
fam["mean_ablation_rule"] = [bool(mean_rule.get((r.model, r.hypothesis), False)) for r in fam.itertuples()]
fam["robust"] = fam.supported & fam.mean_ablation_rule
fam["_m"] = fam.model.map({m: i for i, m in enumerate(CONFIG["MODELS"])})
fam = fam.sort_values(["_m", "hypothesis"]).drop(columns="_m").reset_index(drop=True)

# =============================================================================
# 6. Save, table, figure
# =============================================================================
desc = pd.DataFrame(desc_rows)
sel = pd.DataFrame(sel_rows)
hyp.to_csv(RES / f"hypothesis_contrasts{sfx}.csv", index=False)
fam.to_csv(RES / f"secondary_family_holm{sfx}.csv", index=False)
desc.to_csv(RES / f"measures{sfx}.csv", index=False)
sel.to_csv(RES / f"probe_selectivity{sfx}.csv", index=False)
C.save_json(dict(plan_sha256=FROZEN["plan_sha256"], lesion_sets_sha256=sets_sha, holdout_fingerprint=spec["fingerprint"],
                 config=CONFIG, n_holdout_objects=len(HO.objects), n_hue_objects=int(hue_obj_ok.sum()),
                 n_pairs=len(PAIR_IDS), n_reference_pairs=len(REF_KEYS), secondary_family=fam.to_dict(orient="records"),
                 run_at=time.strftime("%Y-%m-%d %H:%M:%S")), RES / f"stage6_summary{sfx}.json")

lines = [r"\begin{tabular}{lllrrl}", r"\hline",
         r"Model & Hypothesis & Contrast (configural minus damage-matched random) & Estimate [95\% CI] & $p_\mathrm{Holm}$ & Supported \\",
         r"\hline"]
names = {"H4 hue": "hue probe drop", "H4 catscr": "scrambled-category probe drop", "H5": "change in JS distance to BagNet-33",
         "H6": "change in edge minus silhouette accuracy", "H7": "change in overlap cost"}
for r in hyp[hyp.ablation == "zero"].itertuples():
    f = fam[(fam.model == r.model) & (fam.hypothesis == r.hypothesis)].iloc[0]
    lines.append(f"{C.MODEL_INFO[r.model][0]} & {r.hypothesis} & {names[r.contrast]} & "
                 f"{r.estimate:0.3f} [{r.lo:0.3f}, {r.hi:0.3f}] & {f.p_holm:0.4f} & {'yes' if f.supported else 'no'} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage6_secondary_table.tex").write_text("\n".join(lines))


def tidy(ax, decimals):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{decimals}f}"))


def nice(lo, hi, steps=(0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)):
    span = max(hi - lo, 1e-9)
    step = next((s for s in steps if span / s <= 6), steps[-1])
    return np.floor(lo / step) * step, np.ceil(hi / step) * step, step


def figure():
    """Each measure: intact, configural lesion and damage-matched random lesions (mean, 95% range of sets)."""
    C.set_pub_style()
    panels = [("probe accuracy: hue", "Hue probe accuracy\n(scrambled images)"),
              ("probe accuracy: catscr", "Category probe accuracy\n(scrambled images)"),
              ("JS distance to BagNet-33 errors", "Distance to BagNet-33\nerrors (JS)"),
              ("edge_minus_sil", "Edge minus silhouette\naccuracy"),
              ("overlap_cost", "Overlap cost\n(side by side minus overlap)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.6))
    for p, (ax, (meas, ylab)) in enumerate(zip(axes, panels)):
        d = desc[(desc.measure == meas) & (desc.ablation == "zero")]
        vals = [0.0]
        for i, m in enumerate(CONFIG["MODELS"]):
            r = d[d.model == m].iloc[0]
            ax.plot(i - 0.25, r.intact, "o", ms=11, mfc="white", mec="#000000", mew=2, label="Intact" if i == 0 else None)
            ax.plot(i, r.configural, "o", ms=11, color="#2a78d6", label="Configural lesion" if i == 0 else None)
            ax.errorbar(i + 0.25, r.random_mean, yerr=[[r.random_mean - r.random_lo], [r.random_hi - r.random_mean]],
                        fmt="D", ms=9, capsize=5, lw=1.8, color="#8a8a8a",
                        label="Random lesions (mean, 95% of sets)" if i == 0 else None)
            vals += [r.intact, r.configural, r.random_lo, r.random_hi]
        lo, hi, step = nice(min(vals), max(vals))
        dec = 2 if step < 0.1 else 1
        C.style_axes(ax, ylim=(lo, hi), ystep=step, decimals=dec)
        tidy(ax, dec)
        ax.set_xticks(range(len(CONFIG["MODELS"])))
        ax.set_xticklabels([C.MODEL_INFO[m][0].replace(" (", "\n(") for m in CONFIG["MODELS"]])
        ax.set_xlim(-0.6, len(CONFIG["MODELS"]) - 0.4)
        ax.set_ylabel(ylab)
        ax.yaxis.set_label_coords(-0.2, 0.5)
        ax.text(-0.2, 1.03, "abcde"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.55, bottom=0.25)
    C.save_figure(fig, FIGS / f"stage6_signatures{sfx}")
    plt.close(fig)


figure()

pd.set_option("display.width", 240)
show = lambda df: df.round(4).astype(object).where(df.notna(), "").to_string(index=False)
print("\nMeasures: intact, configural lesion, damage-matched random lesions (mean and 95% range of sets)")
print(show(desc))
print("\nProbe selectivity (calibration, held-out scramble variant; control = labels shuffled across objects)")
print(show(sel))
print("\nContrasts: configural minus damage-matched random")
print(show(hyp))
print("\nSecondary family, Holm-adjusted (8 tests; zero ablation decides, mean ablation for robustness)")
print(show(fam))
print("\n" + "=" * 78)
for r in fam.itertuples():
    print(f"  {C.MODEL_INFO[r.model][0]:<18} {r.hypothesis}: {'SUPPORTED' if r.supported else 'not supported'}"
          f" | Holm p {r.p_holm:0.4f} | rule met: {r.rule_met} | mean ablation agrees: {r.mean_ablation_rule}")
print("=" * 78)
print("\nDone. Outputs in", RES, "and", FIGS)
