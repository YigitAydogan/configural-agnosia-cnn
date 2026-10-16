"""
STAGE 5: Confirmatory lesion tests on the hold-out set (H1 to H3)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 1.5 to 2 hours on an A100, mostly
CORnet-S). Every model and lesion block is saved as soon as it finishes; if the runtime disconnects,
rerunning picks up where it stopped. The statistics are recomputed from the saved decisions.

This is the first stage that evaluates the study models on the hold-out set. Before any hold-out image
is used, the script checks that:
  - the analysis plan is frozen (plan SHA-256 dc1a1b0e...),
  - Amendment A1 on Drive matches its registered SHA-256 (6dd4a744...),
  - lesion_sets_A1.json matches the SHA-256 published in release preregistration-v1.2 (9bb05eb1...),
  - the hold-out images still match the recorded fingerprint.
Every model run on hold-out images is written to plan/holdout_access_log.jsonl.

What it runs (everything is fixed by the frozen plan and A1; nothing is chosen here)
  Data      hold-out textured objects: 12 px grid control and its scrambles (layout change >= 10%);
            categories excluded from primary means: knife (no hold-out objects) and any with < 20 objects.
  Outcome   CR = (A_grid - A_scr) / (A_grid - 1/16), 16-way accuracy, category-balanced; floor rule:
            CR not computed when the lower 95% bound of A_grid is below 0.125.
  H1        CR(model) - CR(BagNet-33), intact: lower 95% bound > 0.
  H2        dCR(configural) - mean dCR(random) for size-matched and damage-matched sets: both upper bounds
            < 0, and dCR(configural) below the 5th percentile of the 100 damage-matched random lesions.
  H3        dCR(configural) - dCR(local): upper bound < 0.
  Decision  The rule above AND a Holm-adjusted two-sided bootstrap p < 0.05 within the primary family
            (H1 to H3 x 2 models = 6 tests). For H2 the p-value is the larger of its two contrasts
            (intersection-union test). A hypothesis whose configural lesion is at floor is not supported.
  Robustness  H2 and H3 repeated with mean ablation (channel mean over calibration images, stored in
            lesion_sets_A1.json); robust only if zero and mean ablation agree in direction and significance.
  Sensitivity absolute scramble loss (A_grid - A_scr), source-weighted means (every object equal),
            all other reliable sites (configural set vs 20 size-matched random sets).
  Secondary dose-response at the primary site: 1, 2, 5, 10, 20, 40% of channels by I_CR vs 20 size-matched
            random sets per size.
  Bootstrap category-stratified over objects, 10,000 resamples, all conditions resampled jointly (paired).

One rule the plan leaves open, fixed here before any hold-out result is seen: a random control set whose
own A_grid is at floor has no CR, so it is left out of the random mean and the percentile test, and the
number left out is reported. A sensitivity line recomputes H2 with those sets' CR kept (A_grid > 1/16).

Outputs: results/stage5/*.csv|json|tex, figures/stage5/*.pdf|png
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
import matplotlib.pyplot as plt
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
FAKE = SMOKE and os.environ.get("AGNOSIA_FAKE") is not None   # offline test of the statistics with synthetic decisions
sfx = ("_smoke" if SMOKE else "") + ("_fake" if FAKE else "")

PLAN_SHA_REGISTERED = "dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c"
AMENDMENT_SHA = "6dd4a7445862ba90f392a6028623abdf4d98340ada89fa8ff6f396bc5b814e9f"
SETS_SHA_REGISTERED = "9bb05eb1ee8729786246ff80f23575225c88f3271c4c6756a92525500c2fb904"

assert C.FROZEN_PLAN.exists(), "The analysis plan is not frozen."
FROZEN = C.load_json(C.FROZEN_PLAN)
PLAN = FROZEN["plan"]
ok, why = C.holdout_status()
assert ok, f"Hold-out locked: {why}"
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{'_smoke' if SMOKE else ''}.json"
AMEND_PATH = C.PLAN_DIR / "amendment_01.json"
sets_sha = hashlib.sha256(SETS_PATH.read_bytes()).hexdigest()
if not SMOKE:
    assert FROZEN["plan_sha256"] == PLAN_SHA_REGISTERED, "Frozen plan differs from the registered plan."
    assert hashlib.sha256(AMEND_PATH.read_bytes()).hexdigest() == AMENDMENT_SHA, "Amendment differs from the registered one."
    assert sets_sha == SETS_SHA_REGISTERED, "lesion_sets_A1.json differs from the version published in v1.2."
SETS = C.load_json(SETS_PATH)

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    REFERENCE=PLAN["models"]["local_reference"],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    FLOOR=2 * C.CHANCE_16,
    N_BOOT=10000,
    ALPHA=0.05,
    BATCH=128,
    MAX_SUFFIX_IMAGES=1024,
    MAX_ELEMENTS=4e8,
)
if SMOKE:
    CONFIG.update(N_BOOT=500, BATCH=32, MIN_OBJECTS_PER_CATEGORY=2)

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True     # same numerical settings as Stage 3, where the sets were chosen
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage5"
CKPT = RES / "checkpoints"
FIGS = C.FIG_DIR / "stage5"
for d in [RES, CKPT, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
print(f"Stage 5 (confirmatory) | plan {FROZEN['plan_sha256'][:12]} | amendment {AMENDMENT_SHA[:12]} | "
      f"lesion sets {sets_sha[:12]} | device {DEVICE}")

# =============================================================================
# 1. Hold-out images
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli(C.HOLDOUT_STAGE)
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
spec = C.load_json(C.HOLDOUT_SPEC)
t0 = time.time()
assert C.fingerprint_files(C.STIM_LOCAL, hm.path.tolist()) == spec["fingerprint"], "Hold-out files changed since locking."
print(f"  hold-out fingerprint verified ({len(hm)} files, {time.time() - t0:0.0f} s)")

d = hm[(hm.battery == "configural_tex") & hm.condition.isin(["cutG_intact", "scrambled"])].copy()
d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_STRENGTH"]))]
d = d[~d.category.isin(CONFIG["EXCLUDED"])]
keep = sorted(set(d[d.condition == "cutG_intact"].source_id) & set(d[d.condition == "scrambled"].source_id))
d = d[d.source_id.isin(keep)]
n_per_cat = d.drop_duplicates("source_id").category.value_counts()
small = sorted(n_per_cat[n_per_cat < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)
d = d[~d.category.isin(small)].sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
OBJECTS = sorted(d.source_id.unique())
oidx = {o: i for i, o in enumerate(OBJECTS)}
obj = d.source_id.map(oidx).to_numpy()
is_grid = (d.condition == "cutG_intact").to_numpy()
y = d.category.map(C.CAT_TO_IDX).to_numpy()
obj_cat = d.drop_duplicates("source_id").set_index("source_id").loc[OBJECTS, "category"].to_numpy()
CATS = sorted(set(obj_cat))
grid_of_obj = np.array([np.where(is_grid & (obj == i))[0][0] for i in range(len(OBJECTS))])
SCR = np.zeros((len(d), len(OBJECTS)), np.float32)
for i in range(len(OBJECTS)):
    idx = np.where(~is_grid & (obj == i))[0]
    SCR[idx, i] = 1.0 / len(idx)
size = 64 if SMOKE else C.IMG_SIZE
t0 = time.time()
IMGS = torch.from_numpy(np.stack([np.array(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)), dtype=np.uint8)
                                  for p in d.path])).permute(0, 3, 1, 2).contiguous()
PATHS = d.path.tolist()
print(f"  hold-out: {len(OBJECTS)} objects in {len(CATS)} categories, {len(d)} images "
      f"({is_grid.sum()} grid, {(~is_grid).sum()} scrambled) loaded in {time.time() - t0:0.0f} s"
      + (f" | excluded categories: {', '.join(CONFIG['EXCLUDED'] + small)}" if CONFIG["EXCLUDED"] or small else ""))

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def norm(x_uint8):
    return (x_uint8.to(DEVICE, non_blocking=True).float() / 255.0 - MEAN) / STD


# =============================================================================
# 2. Models
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
            stem = nn.Sequential(m.conv1, m.conv2, m.bn1, m.relu)
            self.stages = [("stem", stem), ("layer1", m.layer1), ("layer2", m.layer2),
                           ("layer3", m.layer3), ("layer4", m.layer4)]
            self.head = lambda x: m.fc(x.mean(dim=(2, 3)))
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
def eval_sets(net, site, sets, mode="zero", mean_act=None):
    """(n_sets + 1, N_images) correct 16-way decisions; row 0 = no lesion. Several sets share one suffix
    call whose copy 0 is the unlesioned network (as in Stage 3)."""
    y_all = torch.from_numpy(y).to(DEVICE)
    out = np.zeros((len(sets) + 1, len(d)), bool)
    mvals = None if mean_act is None else torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
    B = CONFIG["BATCH"]
    for b0 in range(0, len(d), B):
        a = net.prefix(norm(IMGS[b0:b0 + B]), site)
        nb = a.shape[0]
        K = int(max(1, min(CONFIG["MAX_SUFFIX_IMAGES"] // nb - 1, CONFIG["MAX_ELEMENTS"] // a.numel() - 1)))
        yb = y_all[b0:b0 + nb]
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
            logits = net.suffix(aa.flatten(0, 1), site)
            pred = (torch.softmax(logits.float(), 1) @ M16).argmax(1).view(len(chunk) + 1, nb)
            corr = (pred == yb[None, :]).cpu().numpy()
            if s0 == 0:
                out[0, b0:b0 + nb] = corr[0]
            out[1 + s0:1 + s0 + len(chunk), b0:b0 + nb] = corr[1:]
    return out


@torch.no_grad()
def eval_intact(net):
    y_all = torch.from_numpy(y).to(DEVICE)
    out = np.zeros(len(d), bool)
    last = net.stages[-1][0]
    for b0 in range(0, len(d), CONFIG["BATCH"]):
        logits = net.suffix(net.prefix(norm(IMGS[b0:b0 + CONFIG["BATCH"]]), last), last)
        out[b0:b0 + CONFIG["BATCH"]] = ((torch.softmax(logits.float(), 1) @ M16).argmax(1)
                                         == y_all[b0:b0 + CONFIG["BATCH"]]).cpu().numpy()
    return out


# =============================================================================
# 3. Run (checkpointed per block)
# =============================================================================
def blocks_for(name):
    """Every lesion block for a model: (block id, site, mode, labels, sets)."""
    mo = SETS["models"][name]
    ps = mo["primary_site"]
    e = mo["sites"][ps]
    out = []
    labels = ["configural", "local"] + (["selective"] if len(e.get("selective", [])) else [])
    sets = [e["configural"], e["local"]] + ([e["selective"]] if len(e.get("selective", [])) else [])
    labels += [f"size {i + 1}" for i in range(len(e["random_size_matched"]))]
    sets += e["random_size_matched"]
    labels += [f"damage {i + 1}" for i in range(len(e["random_damage_matched"]))]
    sets += e["random_damage_matched"]
    out.append(("primary_zero", ps, "zero", labels, sets))
    nm = 2 + len(e["random_size_matched"]) + len(e["random_damage_matched"])
    ml = [l for l in labels if l != "selective"]
    ms = [s for l, s in zip(labels, sets) if l != "selective"]
    out.append(("primary_mean", ps, "mean", ml, ms))
    dl, ds = [], []
    for f, v in e["dose_response"].items():
        dl.append(f"dose {f} configural")
        ds.append(v["configural"])
        dl += [f"dose {f} random {i + 1}" for i in range(len(v["random"]))]
        ds += v["random"]
    out.append(("dose", ps, "zero", dl, ds))
    for s, es in mo["sites"].items():
        if s == ps or not es.get("reliable") or "random_size_matched" not in es:
            continue
        out.append((f"site_{s}", s, "zero", ["configural"] + [f"size {i + 1}" for i in range(len(es["random_size_matched"]))],
                    [es["configural"]] + es["random_size_matched"]))
    return out


def fake_rows(name, labels):
    """Synthetic decisions with known effects (offline test only): configural lesions remove most of the
    scramble cost, random and local lesions remove grid accuracy but keep it; BagNet is scramble-insensitive."""
    rng = np.random.default_rng(C.derive_seed("fake", name, len(labels)))
    out = []
    for l in labels:
        if name == CONFIG["REFERENCE"]:
            pg, ps = 0.50, 0.47
        elif l == "intact":
            pg, ps = 0.80, 0.40
        elif "configural" in l or l == "selective":
            pg, ps = 0.60, 0.55
        elif l == "local":
            pg, ps = 0.60, 0.30
        else:
            pg, ps = 0.60 + rng.normal(0, 0.03), 0.33 + rng.normal(0, 0.03)
        p = np.where(is_grid, pg, ps)
        out.append(rng.random(len(d)) < p)
    return np.array(out)


RESULTS = {}         # model -> block id -> (labels, correctness rows incl. row 0 'intact')
for name in CONFIG["MODELS"] + [CONFIG["REFERENCE"]]:
    print(f"\n[{C.MODEL_INFO[name][0]}]")
    net = None
    RESULTS[name] = {}
    plan_blocks = [("intact", None, None, [], [])] if name == CONFIG["REFERENCE"] else blocks_for(name)
    for bid, site, mode, labels, sets in plan_blocks:
        ck = CKPT / f"{name}_{bid}{sfx}.npz"
        if ck.exists():
            z = np.load(ck, allow_pickle=True)
            RESULTS[name][bid] = (list(z["labels"]), z["correct"])
            print(f"  {bid}: loaded from checkpoint")
            continue
        if net is None and not FAKE:
            C.check_holdout_access(PATHS, name)        # logs the run; refuses if the plan is not frozen
            net = Split(name)
        t0 = time.time()
        if bid == "intact":
            corr = fake_rows(name, ["intact"]) if FAKE else eval_intact(net)[None]
        else:
            mean_act = SETS["models"][name]["sites"][site]["mean_activation"] if mode == "mean" else None
            corr = fake_rows(name, ["intact"] + labels) if FAKE else eval_sets(net, site, sets, mode=mode, mean_act=mean_act)
        np.savez_compressed(ck, labels=np.array(["intact"] + labels, dtype=object), correct=corr)
        RESULTS[name][bid] = (["intact"] + labels, corr)
        print(f"  {bid} ({site}, {mode} ablation): {len(sets)} lesion sets x {len(d)} images in "
              f"{(time.time() - t0) / 60:0.1f} min")
    del net
    torch.cuda.empty_cache()

# =============================================================================
# 4. Bootstrap (category-stratified over objects, all conditions jointly)
# =============================================================================
def weights(balanced=True):
    """Point weights and (N_BOOT, n_objects) resample weights for category-balanced or object-weighted means."""
    rng = np.random.default_rng(C.derive_seed("stage5_bootstrap"))
    n_obj = len(OBJECTS)
    W = np.zeros((CONFIG["N_BOOT"], n_obj), np.float32)
    w0 = np.zeros(n_obj, np.float32)
    groups = [np.where(obj_cat == c)[0] for c in CATS]
    for g in groups:
        scale = 1.0 / (len(g) * len(CATS)) if balanced else 1.0 / n_obj
        w0[g] = scale
        draws = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
        cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
        np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), draws.ravel()), 1)
        W[:, g] = cnt * scale
    return w0, W


WEIGHTS = {"balanced": weights(True), "source": weights(False)}


def accuracies(corr, wkey="balanced"):
    """corr: (n_cond, N_images) -> point and bootstrap A_grid, A_scr: arrays (n_cond,), (N_BOOT, n_cond)."""
    w0, W = WEIGHTS[wkey]
    G = corr[:, grid_of_obj].T.astype(np.float32)          # (n_obj, n_cond)
    S = (corr.astype(np.float32) @ SCR).T                   # (n_obj, n_cond)
    return w0 @ G, w0 @ S, W @ G, W @ S


def cr_from(ag, asc):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ag - asc) / (ag - C.CHANCE_16)


class Cond:
    """Point and bootstrap CR, absolute loss and floor status for a correctness block."""

    def __init__(self, labels, corr, wkey="balanced"):
        self.labels = labels
        self.ix = {l: i for i, l in enumerate(labels)}
        ag, asc, AG, AS = accuracies(corr, wkey)
        self.ag, self.asc, self.AG, self.AS = ag, asc, AG, AS
        self.cr, self.CR = cr_from(ag, asc), cr_from(AG, AS)
        self.loss, self.LOSS = ag - asc, AG - AS
        self.ag_lo = np.quantile(AG, 0.025, axis=0)
        self.floor = self.ag_lo < CONFIG["FLOOR"]

    def get(self, label, measure="cr"):
        i = self.ix[label]
        return (self.cr[i], self.CR[:, i]) if measure == "cr" else (self.loss[i], self.LOSS[:, i])


def summarize(point, boot):
    boot = np.asarray(boot, float)
    ok = np.isfinite(boot)
    lo, hi = (np.quantile(boot[ok], [0.025, 0.975]) if ok.sum() > 10 else (np.nan, np.nan))
    n = ok.sum()
    p = min(1.0, 2 * min((1 + (boot[ok] <= 0).sum()) / (n + 1), (1 + (boot[ok] >= 0).sum()) / (n + 1))) if n else 1.0
    return dict(estimate=float(point), lo=float(lo), hi=float(hi), p_two_sided=float(p))


def h2_h3(cond, measure="cr", keep_floor_random=False):
    """H2 (size- and damage-matched) and H3 statistics from one primary block."""
    if measure == "cr" and cond.floor[cond.ix["configural"]]:
        return None
    base_p, base_b = cond.get("intact", measure)
    c_p, c_b = cond.get("configural", measure)
    dconf_p, dconf_b = c_p - base_p, c_b - base_b
    out = {}
    for kind in ["size", "damage"]:
        labs = [l for l in cond.labels if l.startswith(kind + " ")]
        if measure != "cr":
            use = labs                                                     # no floor rule for absolute loss
        elif keep_floor_random:
            use = [l for l in labs if cond.ag[cond.ix[l]] > C.CHANCE_16]   # CR defined, floor ignored
        else:
            use = [l for l in labs if not cond.floor[cond.ix[l]]]          # pre-stated rule
        ii = [cond.ix[l] for l in use]
        vals_p = (cond.cr if measure == "cr" else cond.loss)[ii] - base_p
        vals_b = (cond.CR if measure == "cr" else cond.LOSS)[:, ii] - base_b[:, None]
        r = summarize(dconf_p - vals_p.mean(), dconf_b - np.nanmean(vals_b, 1))
        r.update(n_random=len(labs), n_random_used=len(use), random_mean_dCR=float(vals_p.mean()))
        if kind == "damage":
            r["random_5th_percentile"] = float(np.percentile(vals_p, 5))
            r["below_5th_percentile"] = bool(dconf_p < np.percentile(vals_p, 5))
        out[kind] = r
    if "local" in cond.ix and not (measure == "cr" and cond.floor[cond.ix["local"]]):
        l_p, l_b = cond.get("local", measure)
        out["local"] = summarize(dconf_p - (l_p - base_p), dconf_b - (l_b - base_b))
    else:
        out["local"] = None
    out["dCR_configural"] = summarize(dconf_p, dconf_b)
    return out


# =============================================================================
# 5. Hypotheses
# =============================================================================
ref_block = Cond(*RESULTS[CONFIG["REFERENCE"]]["intact"])
rows, robust_rows, sens_rows, cond_rows, dose_rows, site_rows = [], [], [], [], [], []
for name in CONFIG["MODELS"]:
    label = C.MODEL_INFO[name][0]
    ps = SETS["models"][name]["primary_site"]
    blk = Cond(*RESULTS[name]["primary_zero"])
    # condition table
    for l in blk.labels:
        i = blk.ix[l]
        cond_rows.append(dict(model=name, block="primary_zero", condition=l, A_grid=blk.ag[i], A_grid_lo=blk.ag_lo[i],
                              A_scr=blk.asc[i], CR=np.nan if blk.floor[i] else blk.cr[i], floor=bool(blk.floor[i]),
                              loss=blk.loss[i]))
    # H1: intact CR vs BagNet-33 on the same objects (the intact rows of every block are identical)
    m_p, m_b = blk.get("intact")
    r_p, r_b = ref_block.get("intact")
    h1 = summarize(m_p - r_p, m_b - r_b)
    rows.append(dict(model=name, hypothesis="H1", contrast="CR(model) - CR(BagNet-33)", **h1,
                     rule_met=bool(h1["lo"] > 0), floor=False))
    # H2 and H3
    st = h2_h3(blk)
    if st is None:
        rows.append(dict(model=name, hypothesis="H2", contrast="configural lesion at floor", estimate=np.nan, lo=np.nan,
                         hi=np.nan, p_two_sided=1.0, rule_met=False, floor=True))
        rows.append(dict(model=name, hypothesis="H3", contrast="configural lesion at floor", estimate=np.nan, lo=np.nan,
                         hi=np.nan, p_two_sided=1.0, rule_met=False, floor=True))
    else:
        sz, dm = st["size"], st["damage"]
        rule2 = bool(sz["hi"] < 0 and dm["hi"] < 0 and dm["below_5th_percentile"])
        rows.append(dict(model=name, hypothesis="H2", contrast="dCR(config) - mean dCR(size-matched random)", **sz,
                         rule_met=rule2, floor=False, p_iut=max(sz["p_two_sided"], dm["p_two_sided"])))
        rows.append(dict(model=name, hypothesis="H2", contrast="dCR(config) - mean dCR(damage-matched random)", **dm,
                         rule_met=rule2, floor=False, p_iut=max(sz["p_two_sided"], dm["p_two_sided"])))
        lo_ = st["local"]
        if lo_ is None:
            rows.append(dict(model=name, hypothesis="H3", contrast="local lesion at floor", estimate=np.nan, lo=np.nan,
                             hi=np.nan, p_two_sided=1.0, rule_met=False, floor=True))
        else:
            rows.append(dict(model=name, hypothesis="H3", contrast="dCR(config) - dCR(local)", **lo_,
                             rule_met=bool(lo_["hi"] < 0), floor=False))
    # robustness: mean ablation
    blk_m = Cond(*RESULTS[name]["primary_mean"])
    st_m = h2_h3(blk_m)
    for l in blk_m.labels:
        i = blk_m.ix[l]
        cond_rows.append(dict(model=name, block="primary_mean", condition=l, A_grid=blk_m.ag[i], A_grid_lo=blk_m.ag_lo[i],
                              A_scr=blk_m.asc[i], CR=np.nan if blk_m.floor[i] else blk_m.cr[i],
                              floor=bool(blk_m.floor[i]), loss=blk_m.loss[i]))
    for hyp_, key in [("H2", "size"), ("H2", "damage"), ("H3", "local")]:
        v = None if st_m is None else st_m[key]
        robust_rows.append(dict(model=name, hypothesis=hyp_, control=key, ablation="mean",
                                configural_floor=st_m is None,
                                **(v if v else dict(estimate=np.nan, lo=np.nan, hi=np.nan, p_two_sided=1.0))))
    # sensitivity: absolute loss, source weighting, floor random sets kept
    for tag, cnd, meas, keepf in [("absolute scramble loss", blk, "loss", False),
                                  ("source-weighted", Cond(*RESULTS[name]["primary_zero"], wkey="source"), "cr", False),
                                  ("random sets at floor kept", blk, "cr", True)]:
        s2 = h2_h3(cnd, meas, keepf)
        for key in ["size", "damage", "local"]:
            v = None if s2 is None else s2[key]
            sens_rows.append(dict(model=name, analysis=tag, control=key,
                                  **(v if v else dict(estimate=np.nan, lo=np.nan, hi=np.nan, p_two_sided=np.nan))))
    if name:
        src = Cond(*RESULTS[name]["primary_zero"], wkey="source")
        sp, sb = src.get("intact")
        rp, rb = Cond(*RESULTS[CONFIG["REFERENCE"]]["intact"], wkey="source").get("intact")
        sens_rows.append(dict(model=name, analysis="source-weighted", control="H1 vs BagNet-33", **summarize(sp - rp, sb - rb)))
        lp, lb = blk.get("intact", "loss")
        rlp, rlb = ref_block.get("intact", "loss")
        sens_rows.append(dict(model=name, analysis="absolute scramble loss", control="H1 vs BagNet-33",
                              **summarize(lp - rlp, lb - rlb)))
    # dose-response
    dz = Cond(*RESULTS[name]["dose"])
    base_p, base_b = dz.get("intact")
    for f in SETS["models"][name]["sites"][ps]["dose_response"]:
        ci = dz.ix[f"dose {f} configural"]
        rl = [l for l in dz.labels if l.startswith(f"dose {f} random")]
        rv = np.array([np.nan if dz.floor[dz.ix[l]] else dz.cr[dz.ix[l]] - base_p for l in rl])
        rb = np.stack([dz.CR[:, dz.ix[l]] - base_b for l in rl if not dz.floor[dz.ix[l]]], 1) if np.isfinite(rv).any() else None
        cfl = bool(dz.floor[ci])
        row = dict(model=name, site=ps, fraction=float(f), n_channels=len(SETS["models"][name]["sites"][ps]["dose_response"][f]["configural"]),
                   dCR_configural=np.nan if cfl else dz.cr[ci] - base_p, configural_floor=cfl,
                   dCR_random_mean=float(np.nanmean(rv)) if np.isfinite(rv).any() else np.nan,
                   dCR_random_min=float(np.nanmin(rv)) if np.isfinite(rv).any() else np.nan,
                   dCR_random_max=float(np.nanmax(rv)) if np.isfinite(rv).any() else np.nan,
                   random_at_floor=int((~np.isfinite(rv)).sum()), n_random=len(rl),
                   A_grid_configural=float(dz.ag[ci]), A_grid_configural_lo=float(dz.ag_lo[ci]),
                   A_scr_configural=float(dz.asc[ci]), loss_configural=float(dz.loss[ci]),
                   A_grid_random_mean=float(np.mean([dz.ag[dz.ix[l]] for l in rl])),
                   A_grid_intact=float(dz.ag[0]))
        for l in ["intact", f"dose {f} configural"] + rl:
            i = dz.ix[l]
            if l == "intact" and f != list(SETS["models"][name]["sites"][ps]["dose_response"])[0]:
                continue
            cond_rows.append(dict(model=name, block="dose", condition=l, A_grid=dz.ag[i], A_grid_lo=dz.ag_lo[i],
                                  A_scr=dz.asc[i], CR=np.nan if dz.floor[i] else dz.cr[i], floor=bool(dz.floor[i]),
                                  loss=dz.loss[i]))
        if not cfl:
            s = summarize(dz.cr[ci] - base_p, dz.CR[:, ci] - base_b)
            row.update(dCR_configural_lo=s["lo"], dCR_configural_hi=s["hi"])
            if rb is not None:
                s2 = summarize(row["dCR_configural"] - row["dCR_random_mean"], (dz.CR[:, ci] - base_b) - np.nanmean(rb, 1))
                row.update(diff=s2["estimate"], diff_lo=s2["lo"], diff_hi=s2["hi"])
        dose_rows.append(row)
    # other reliable sites
    for bid in [b for b in RESULTS[name] if b.startswith("site_")]:
        sb_ = Cond(*RESULTS[name][bid])
        bp, bb = sb_.get("intact")
        ci = sb_.ix["configural"]
        rl = [l for l in sb_.labels if l.startswith("size ")]
        use = [l for l in rl if not sb_.floor[sb_.ix[l]]]
        row = dict(model=name, site=bid[5:], n_channels=len(SETS["models"][name]["sites"][bid[5:]]["configural"]),
                   configural_floor=bool(sb_.floor[ci]), n_random=len(rl), n_random_used=len(use))
        if not sb_.floor[ci] and use:
            ii = [sb_.ix[l] for l in use]
            s = summarize((sb_.cr[ci] - bp) - np.mean(sb_.cr[ii] - bp),
                          (sb_.CR[:, ci] - bb) - np.nanmean(sb_.CR[:, ii] - bb[:, None], 1))
            row.update(dCR_configural=sb_.cr[ci] - bp, **{f"diff_{k}": v for k, v in s.items()})
        site_rows.append(row)

# Holm within the primary family (6 tests)
hyp = pd.DataFrame(rows)
fam = []
for (m, h), g in hyp.groupby(["model", "hypothesis"], sort=False):
    p = g["p_iut"].iloc[0] if h == "H2" and "p_iut" in g and np.isfinite(g["p_iut"].iloc[0]) else g["p_two_sided"].iloc[0]
    fam.append(dict(model=m, hypothesis=h, p=float(p), rule_met=bool(g.rule_met.iloc[0]), floor=bool(g.floor.iloc[0])))
fam = pd.DataFrame(fam).sort_values("p").reset_index(drop=True)
k = len(fam)
adj, run = [], 0.0
for i, p in enumerate(fam.p):
    run = max(run, min(1.0, (k - i) * p))
    adj.append(run)
fam["p_holm"] = adj
fam["supported"] = fam.rule_met & (fam.p_holm < CONFIG["ALPHA"]) & ~fam.floor
fam["_m"] = fam.model.map({m: i for i, m in enumerate(CONFIG["MODELS"])})
fam = fam.sort_values(["_m", "hypothesis"]).drop(columns="_m").reset_index(drop=True)

# robustness verdict: zero and mean ablation agree in direction and significance (unadjusted 95% bounds)
rob = pd.DataFrame(robust_rows)
verdicts = []
for (m, h), g in rob.groupby(["model", "hypothesis"]):
    zero = hyp[(hyp.model == m) & (hyp.hypothesis == h)]
    zero_sig = bool(zero.rule_met.iloc[0]) and not bool(zero.floor.iloc[0])
    mean_sig = bool((g.hi < 0).all()) and not bool(g.configural_floor.any())
    if h == "H2":
        dmg = g[g.control == "damage"]
        mean_sig = mean_sig and bool(dmg.get("below_5th_percentile", pd.Series([False])).fillna(False).iloc[0])
    verdicts.append(dict(model=m, hypothesis=h, zero_ablation_rule=zero_sig, mean_ablation_rule=mean_sig,
                         robust=zero_sig and mean_sig))
rob_v = pd.DataFrame(verdicts)
rob_v["_m"] = rob_v.model.map({m: i for i, m in enumerate(CONFIG["MODELS"])})
rob_v = rob_v.sort_values(["_m", "hypothesis"]).drop(columns="_m").reset_index(drop=True)

# =============================================================================
# 6. Save
# =============================================================================
pd.DataFrame(cond_rows).to_csv(RES / f"conditions{sfx}.csv", index=False)
hyp.to_csv(RES / f"hypothesis_contrasts{sfx}.csv", index=False)
fam.to_csv(RES / f"primary_family_holm{sfx}.csv", index=False)
rob.to_csv(RES / f"robustness_mean_ablation{sfx}.csv", index=False)
rob_v.to_csv(RES / f"robustness_verdicts{sfx}.csv", index=False)
pd.DataFrame(sens_rows).to_csv(RES / f"sensitivity{sfx}.csv", index=False)
dose_df = pd.DataFrame(dose_rows)
dose_df.to_csv(RES / f"dose_response{sfx}.csv", index=False)
pd.DataFrame(site_rows).to_csv(RES / f"other_sites{sfx}.csv", index=False)
C.save_json(dict(plan_sha256=FROZEN["plan_sha256"], amendment_sha256=AMENDMENT_SHA, lesion_sets_sha256=sets_sha,
                 holdout_fingerprint=spec["fingerprint"], n_objects=len(OBJECTS), categories=CATS,
                 excluded_categories=CONFIG["EXCLUDED"] + small, config=CONFIG,
                 primary_family=fam.to_dict(orient="records"),
                 robustness=rob_v.to_dict(orient="records"),
                 reference_CR=float(ref_block.cr[0]), run_at=time.strftime("%Y-%m-%d %H:%M:%S")),
            RES / f"stage5_summary{sfx}.json")

# LaTeX: primary family
fmt = lambda r: "floor" if r.get("floor") else f"{r['estimate']:0.3f} [{r['lo']:0.3f}, {r['hi']:0.3f}]"
lines = [r"\begin{tabular}{lllrrl}", r"\hline",
         r"Model & Hypothesis & Contrast & Estimate [95\% CI] & $p_\mathrm{Holm}$ & Supported \\", r"\hline"]
for r in hyp.to_dict(orient="records"):
    f = fam[(fam.model == r["model"]) & (fam.hypothesis == r["hypothesis"])].iloc[0]
    lines.append(f"{C.MODEL_INFO[r['model']][0]} & {r['hypothesis']} & {r['contrast']} & {fmt(r)} & "
                 f"{f.p_holm:0.4f} & {'yes' if f.supported else 'no'} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage5_primary_table.tex").write_text("\n".join(lines).replace("dCR", r"$\Delta$CR"))

# =============================================================================
# 7. Figure: dose-response (a, b) and the H2 damage-matched distribution (c, d)
# =============================================================================
def nice(lo, hi, steps=(0.05, 0.1, 0.2, 0.25, 0.5, 1.0)):
    span = max(hi - lo, 1e-9)
    step = next((s for s in steps if span / s <= 6), steps[-1])
    return np.floor(lo / step) * step, np.ceil(hi / step) * step, step


def tidy(ax, axis, decimals):
    """Tick labels without negative zero."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{decimals}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def figure():
    C.set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for col, name in enumerate(CONFIG["MODELS"]):
        ax = axes[0, col]
        dd = dose_df[dose_df.model == name].sort_values("fraction")
        x = 100 * dd.fraction.to_numpy()
        ok = np.isfinite(dd.dCR_random_mean.to_numpy())
        ax.fill_between(x[ok], dd.dCR_random_min[ok], dd.dCR_random_max[ok], color="#8a8a8a", alpha=0.3, lw=0,
                        label="Random, range")
        ax.plot(x[ok], dd.dCR_random_mean[ok], color="#8a8a8a", lw=2, marker="D", ms=8, label="Random, mean")
        okc = ~dd.configural_floor.to_numpy()
        if okc.any():
            yv = dd.dCR_configural.to_numpy()[okc]
            ax.errorbar(x[okc], yv, yerr=[yv - dd.dCR_configural_lo.to_numpy()[okc], dd.dCR_configural_hi.to_numpy()[okc] - yv],
                        color="#2a78d6", lw=2, marker="o", ms=9, capsize=5, label="Top channels by configural score")
        for xf in x[~okc]:
            ax.annotate("floor", (xf, 0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=13,
                        color="#2a78d6")
        vals = np.concatenate([dd.dCR_random_min.to_numpy(), dd.dCR_random_max.to_numpy(),
                               dd.get("dCR_configural_lo", pd.Series(dtype=float)).to_numpy(),
                               dd.get("dCR_configural_hi", pd.Series(dtype=float)).to_numpy(), [0.0]])
        vals = vals[np.isfinite(vals)]
        lo, hi, step = nice(vals.min(), vals.max())
        ax.set_xscale("log")
        ax.set_xlim(0.8, 50)
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
        ax.minorticks_off()
        C.style_axes(ax, ylim=(lo, hi), ystep=step, decimals=2 if step < 0.1 else 1)
        tidy(ax, "y", 2 if step < 0.1 else 1)
        ax.axhline(0, color="#000000", lw=0.8)
        ax.set_xlabel(f"Channels lesioned at {SETS['models'][name]['primary_site']} (%)")
        ax.set_ylabel(f"Change in configural reliance\n{C.MODEL_INFO[name][0]}")
        # H2 distribution
        ax = axes[1, col]
        blk = Cond(*RESULTS[name]["primary_zero"])
        bp = blk.cr[0]
        dl = [l for l in blk.labels if l.startswith("damage ")]
        rv = np.array([blk.cr[blk.ix[l]] - bp for l in dl if not blk.floor[blk.ix[l]]])
        cf = None if blk.floor[blk.ix["configural"]] else blk.cr[blk.ix["configural"]] - bp
        allv = np.concatenate([rv, [cf] if cf is not None else [], [0.0]])
        lo, hi, step = nice(allv.min(), allv.max())
        bins = np.linspace(lo, hi, 25)
        if len(rv):
            ax.hist(rv, bins=bins, color="#8a8a8a", edgecolor="#ffffff", label=f"Damage-matched random ({len(rv)})")
            ax.axvline(np.percentile(rv, 5), color="#000000", ls="--", lw=1.5, label="5th percentile")
        if cf is not None:
            ax.axvline(cf, color="#2a78d6", lw=3, label="Configural set (H2)")
        ymax = max(1, np.histogram(rv, bins=bins)[0].max() if len(rv) else 1)
        ystep = max(1, int(np.ceil(ymax / 5)))
        C.style_axes(ax, xlim=(lo, hi), xstep=step, ylim=(0, ystep * int(np.ceil(ymax * 1.15 / ystep))), ystep=ystep,
                     decimals=2 if step < 0.1 else 1)
        tidy(ax, "x", 2 if step < 0.1 else 1)
        tidy(ax, "y", 0)
        ax.set_xlabel("Change in configural reliance")
        ax.set_ylabel(f"Random lesions\n{C.MODEL_INFO[name][0]}")
    for p, ax in enumerate(axes.ravel()):
        ax.yaxis.set_label_coords(-0.14, 0.5)
        ax.text(-0.14, 1.03, "abcd"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    leg = fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    C.outline_legend(leg)
    fig.subplots_adjust(hspace=0.35, wspace=0.35, bottom=0.14)
    C.save_figure(fig, FIGS / f"stage5_lesions{sfx}")
    plt.close(fig)


figure()

# =============================================================================
# 8. Report
# =============================================================================
pd.set_option("display.width", 240)
show = lambda df: df.round(4).astype(object).where(df.notna(), "").to_string(index=False)
print("\nIntact configural reliance (hold-out): " + " | ".join(
    f"{C.MODEL_INFO[n][0]} {Cond(*RESULTS[n]['primary_zero']).cr[0]:0.3f}" for n in CONFIG["MODELS"])
      + f" | {C.MODEL_INFO[CONFIG['REFERENCE']][0]} {ref_block.cr[0]:0.3f}")
print("\nKey conditions at the primary site (zero ablation)")
cd = pd.DataFrame(cond_rows)
print(show(cd[(cd.block == "primary_zero") & cd.condition.isin(["intact", "configural", "local", "selective"])]
           [["model", "condition", "A_grid", "A_grid_lo", "A_scr", "CR", "floor", "loss"]]))
for name in CONFIG["MODELS"]:
    g = cd[(cd.model == name) & (cd.block == "primary_zero")]
    for kind in ["size", "damage"]:
        r = g[g.condition.str.startswith(kind + " ")]
        print(f"  {C.MODEL_INFO[name][0]} {kind}-matched random: CR mean {np.nanmean(r.CR):0.3f} "
              f"[{np.nanmin(r.CR):0.3f}, {np.nanmax(r.CR):0.3f}], {int(r.floor.sum())} of {len(r)} at floor")
print("\nPrimary contrasts (zero ablation)")
print(show(hyp.drop(columns=[c for c in ["p_iut"] if c in hyp])))
print("\nPrimary family, Holm-adjusted (6 tests)")
print(show(fam))
print("\nRobustness: mean ablation")
print(show(rob))
print(show(rob_v))
print("\nSensitivity analyses")
print(show(pd.DataFrame(sens_rows)))
print("\nDose-response (secondary)")
print(show(dose_df))
print("\nDose-response accuracies (configural sets are nested: each larger set contains the smaller ones)")
print(show(dose_df[["model", "fraction", "n_channels", "A_grid_intact", "A_grid_configural", "A_grid_configural_lo",
                    "A_scr_configural", "loss_configural", "configural_floor", "A_grid_random_mean"]]))
print("\nOther reliable sites (sensitivity)")
print(show(pd.DataFrame(site_rows)))
print("\n" + "=" * 78)
for r in fam.itertuples():
    print(f"  {C.MODEL_INFO[r.model][0]:<18} {r.hypothesis}: {'SUPPORTED' if r.supported else 'not supported'}"
          f"{' (configural lesion at floor)' if r.floor else ''} | Holm p {r.p_holm:0.4f} | rule met: {r.rule_met}")
print("=" * 78)
print("\nDone. Outputs in", RES, "and", FIGS)
