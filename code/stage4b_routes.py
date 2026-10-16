"""
STAGE 4b: Two routes to configural reliance (exploratory, calibration test half)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 45 minutes on an A100). Each model's
results are saved as soon as it finishes; if the runtime disconnects, rerunning skips finished models.

Question (from Stage 4a): a network can be sensitive to scrambling in two ways.
  Position-coded parts: each piece adds evidence that depends on where the piece is. The response is
                        additive over pieces, yet scrambling still changes it, because pieces move.
  Interactions:         pieces change each other's contribution (integration proper).
BagNet-33 has neither (its local evidence is additive and position-free), so it is scramble-insensitive.

Decomposition (exact identity, per object). For an arrangement (intact grid or scrambled) with pieces
1..n, v(S) is the category evidence with only the pieces in S present (absent pieces = white background).
Walsh decomposition over the uniform distribution on subsets (as in Stage 4a):
  v(all pieces) = A + I
  A = mean of v over subsets + sum of the first-order (single-piece) Walsh terms  (best additive model)
  I = the remainder, the sum of all interaction terms.
The configural effect on evidence, D = v_grid(all) - v_scrambled(all), therefore splits exactly into
  D = DA + DI,  DA = A_grid - A_scrambled (position-coded parts),  DI = I_grid - I_scrambled (interactions).
The same piece subsets are used in both arrangements (piece k sits in cell k of the grid image and in cell
perm[k] of the scrambled image), so DA and DI are paired within object.

Estimation. A is estimated from 64 sampled subsets per arrangement (32 random subsets and their
complements, with each piece present with probability 0.5) by least squares on +/-1 presence codes, plus
the full image. For the first 24 objects the intact network is also evaluated on every subset, and the
sampled estimate is compared with the exact one (printed as a check).

Conditions (zero ablation at the primary site): intact, configural lesion, local lesion and the first 10
damage-matched random lesions from lesion_sets_A1.json. Reference: BagNet-33, intact.

Expectations written before running (from Stage 4a; exploratory, not a registered test):
  E1  ResNet-50 (SWSL): a substantial share of D is carried by DI, and the configural lesion reduces DI
      more than every damage-matched random lesion.
  E2  CORnet-S: D is carried mainly by DA, and the configural lesion reduces DA more than every random
      lesion, with DI changed no more than by random lesions.
  E3  BagNet-33: D, DA and DI near zero.

Evidence = mean logit of the true category's ImageNet classes minus the mean over the 16 categories.
Nothing touches the hold-out set. Outputs: results/stage4b/*.csv|json|tex, figures/stage4b/*.pdf|png
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
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = False      # full fp32: the components are small differences
torch.backends.cudnn.allow_tf32 = False

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
CONFIG = dict(
    MIN_PIECES=3,
    MIN_PIECE_PX=40,
    N_PAIRS=32,
    N_CHECK=24,
    N_RANDOM=10,
    N_BOOT=10000,
    BATCH=128,
    GRID=3,
    MIN_STRENGTH=0.10,
    REFERENCE="bagnet33",
)
if SMOKE:
    CONFIG.update(N_PAIRS=8, N_CHECK=4, N_RANDOM=3, N_BOOT=300, BATCH=32)

C.ensure_dirs()
C.set_seed()
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage4b"
FIGS = C.FIG_DIR / "stage4b"
CKPT = RES / "checkpoints"
for d in [RES, FIGS, CKPT]:
    d.mkdir(parents=True, exist_ok=True)
sfx = "_smoke" if SMOKE else ""
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{sfx}.json"
assert SETS_PATH.exists(), "lesion_sets_A1.json not found; run Stage 3b first."
SETS = C.load_json(SETS_PATH)
H = W = C.IMG_SIZE
CFG_TAG = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:8]
print(f"Stage 4b (exploratory) | sets {SETS_PATH.name} | amendment {SETS.get('amendment')} | device {DEVICE}")

# =============================================================================
# 1. Calibration test half: grid image, one scrambled image, pieces in both
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("frozen_calib")
    C.unpack_stimuli("stage1")
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
base = fc[(fc.battery == "configural_tex") & (fc.split == "test")]
grid = base[base.condition == "cutG_intact"].drop_duplicates("source_id").set_index("source_id")
scr = base[(base.condition == "scrambled") & (base.manip_strength >= CONFIG["MIN_STRENGTH"])]
scr = scr.sort_values(["source_id", "variant"]).drop_duplicates("source_id").set_index("source_id")   # lowest valid variant
assert not grid.path.map(C.is_holdout_path).any() and not scr.path.map(C.is_holdout_path).any()

masks = {}
for f in sorted((C.STIM_LOCAL / "stage1").glob("masks_*.npz")):
    z = np.load(f)
    masks.update(dict(zip(z["source_ids"].tolist(), z["masks"])))


def grid_layout(mask):
    """All 9 cell rectangles (same geometry as the stimulus generator) and which cells hold a piece."""
    G = CONFIG["GRID"]
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    h, w = r1 - r0, c1 - c0
    ch, cw = min(int(np.ceil(h / G)), H // G), min(int(np.ceil(w / G)), W // G)
    R0 = int(np.clip(r0 - (G * ch - h) // 2, 0, H - G * ch))
    C0 = int(np.clip(c0 - (G * cw - w) // 2, 0, W - G * cw))
    rects = [(R0 + i * ch, R0 + (i + 1) * ch, C0 + j * cw, C0 + (j + 1) * cw) for i in range(G) for j in range(G)]
    filled = [k for k, (y0, y1, x0, x1) in enumerate(rects) if mask[y0:y1, x0:x1].sum() >= CONFIG["MIN_PIECE_PX"]]
    return rects, filled


size = 64 if SMOKE else H
sc_ = size / H
load = lambda p: np.array(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)), dtype=np.uint8)
scale = lambda r: tuple(int(round(v * sc_)) for v in r)
OBJ = []
for sid in sorted(set(grid.index) & set(scr.index)):
    if sid not in masks:
        continue
    rects, filled = grid_layout(masks[sid])
    if len(filled) < CONFIG["MIN_PIECES"]:
        continue
    perm = [int(t) for t in str(scr.loc[sid, "perm"]).split()]
    OBJ.append(dict(source_id=sid, category=grid.loc[sid, "category"],
                    img={"grid": load(grid.loc[sid, "path"]), "scrambled": load(scr.loc[sid, "path"])},
                    cells={"grid": [scale(rects[k]) for k in filled],
                           "scrambled": [scale(rects[perm[k]]) for k in filled]}))
if SMOKE:
    OBJ = OBJ[:16]
n_pieces = np.array([len(o["cells"]["grid"]) for o in OBJ])
obj_cat = np.array([o["category"] for o in OBJ])
print(f"  test half: {len(OBJ)} objects | pieces per object {n_pieces.min()} to {n_pieces.max()} "
      f"(median {int(np.median(n_pieces))}) | {2 * (2 * CONFIG['N_PAIRS'] + 1)} images per object per model")

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def sampled_subsets(o):
    """Presence matrix (rows = subsets): N_PAIRS random subsets, their complements, then the full set."""
    n = len(o["cells"]["grid"])
    rng = np.random.default_rng(C.derive_seed("stage4b_subsets", o["source_id"]))
    P = rng.random((CONFIG["N_PAIRS"], n)) < 0.5
    return np.vstack([P, ~P, np.ones((1, n), bool)])


def all_subsets(n):
    S = np.arange(2 ** n)
    return ((S[:, None] >> np.arange(n)) & 1).astype(bool)


def render(o, arrangement, P):
    """Images of object o in the given arrangement with the pieces in each row of P present."""
    Pt = torch.as_tensor(P, device=DEVICE)
    absent = torch.zeros(len(P), size, size, dtype=torch.bool, device=DEVICE)
    for i, (y0, y1, x0, x1) in enumerate(o["cells"][arrangement]):
        absent[~Pt[:, i], y0:y1, x0:x1] = True
    x = torch.from_numpy(o["img"][arrangement]).to(DEVICE).permute(2, 0, 1)[None].float()
    x = x.expand(len(P), -1, -1, -1).clone()
    x[absent[:, None].expand(-1, 3, -1, -1)] = 255.0
    return (x / 255.0 - MEAN) / STD


def additive_part(P, v):
    """Least squares of v on +/-1 presence codes (with intercept); returns A = prediction at the full set.
    With every subset present (P = all subsets) this is exactly the Walsh truncation to orders 0 and 1."""
    X = np.hstack([np.ones((len(P), 1)), 2.0 * P - 1.0])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return float(beta.sum())          # all codes +1 at the full set


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
        for p in m.parameters():
            p.requires_grad_(False)
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

    def prefix(self, x, site):
        for i in range(self.index[site] + 1):
            x = self.stages[i][1](x)
        return x

    def suffix(self, a, site):
        for i in range(self.index[site] + 1, len(self.stages)):
            a = self.stages[i][1](a)
        return self.head(a)


@torch.no_grad()
def evidence(net, site, x, cat, conditions):
    """Category evidence for every image in x under every condition. Returns dict label -> (len(x),) array."""
    out = {k: [] for k in conditions}
    j = C.CAT_TO_IDX[cat]
    for b0 in range(0, len(x), CONFIG["BATCH"]):
        a = net.prefix(x[b0:b0 + CONFIG["BATCH"]], site)
        for k, ch in conditions.items():
            aa = a
            if len(ch):
                aa = a.clone()
                aa[:, torch.as_tensor(ch, device=DEVICE)] = 0
            e = net.suffix(aa, site).float() @ M16
            out[k].append((e[:, j] - e.mean(1)).double().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


# =============================================================================
# 3. Run (checkpointed per model)
# =============================================================================
def run_model(name, site, conditions):
    ck = CKPT / f"{name}_{CFG_TAG}{sfx}.csv"
    ck_check = CKPT / f"{name}_{CFG_TAG}_check{sfx}.csv"
    if ck.exists() and ck_check.exists():
        print(f"  loaded from checkpoint {ck.name}")
        return pd.read_csv(ck), pd.read_csv(ck_check)
    net = Split(name)
    rows, checks = [], []
    t0 = time.time()
    for oi, o in enumerate(OBJ):
        P = sampled_subsets(o)
        comp = {}
        for arr in ["grid", "scrambled"]:
            ev = evidence(net, site, render(o, arr, P), o["category"], conditions)
            for k, v in ev.items():
                full = v[-1]
                A = additive_part(P[:-1], v[:-1])
                comp[(k, arr)] = (full, A, full - A)
        for k in conditions:
            (vg, Ag, Ig), (vs, As, Is) = comp[(k, "grid")], comp[(k, "scrambled")]
            rows.append(dict(model=name, condition=k, source_id=o["source_id"], category=o["category"],
                             n_pieces=len(P[0]), v_grid=vg, v_scrambled=vs, A_grid=Ag, A_scrambled=As,
                             I_grid=Ig, I_scrambled=Is, D=vg - vs, DA=Ag - As, DI=Ig - Is))
        if oi < CONFIG["N_CHECK"]:                      # exact decomposition, intact network
            Pall = all_subsets(len(P[0]))
            ex = {}
            for arr in ["grid", "scrambled"]:
                v = evidence(net, site, render(o, arr, Pall), o["category"], {"intact": []})["intact"]
                ex[arr] = additive_part(Pall, v)
            r = [r_ for r_ in rows if r_["source_id"] == o["source_id"] and r_["condition"] == "intact"][0]
            checks.append(dict(model=name, source_id=o["source_id"], DA_sampled=r["DA"], DA_exact=ex["grid"] - ex["scrambled"],
                               D=r["D"]))
        if (oi + 1) % 100 == 0 or oi == len(OBJ) - 1:
            el = time.time() - t0
            print(f"    {name} {site}: {oi + 1}/{len(OBJ)} objects | {el / 60:0.1f} min | "
                  f"ETA {el / (oi + 1) * (len(OBJ) - oi - 1) / 60:0.1f} min")
    del net
    torch.cuda.empty_cache()
    df, chk = pd.DataFrame(rows), pd.DataFrame(checks)
    df.to_csv(ck, index=False)
    chk.to_csv(ck_check, index=False)
    return df, chk


def boot(x, rng, strat, stat=np.mean):
    """Category-stratified object bootstrap of stat over rows of x (1-D or 2-D)."""
    x = np.asarray(x, float)
    groups = [np.where(strat == s)[0] for s in np.unique(strat)]
    b = []
    for _ in range(CONFIG["N_BOOT"]):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        b.append(stat(x[idx]))
    b = np.array(b)
    return float(stat(x)), float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))


models = [(m, SETS["models"][m]["primary_site"]) for m in SETS["models"] if SETS["models"][m]["primary_site"]]
models.append((CONFIG["REFERENCE"], "layer4"))
obj_dfs, check_dfs, summ, lesion_rows = [], [], [], []
for name, site in models:
    print(f"\n[{C.MODEL_INFO[name][0]}] site {site}")
    conditions = {"intact": np.array([], int)}
    if name != CONFIG["REFERENCE"]:
        e = SETS["models"][name]["sites"][site]
        conditions["configural lesion"] = np.array(e["configural"], int)
        if len(e.get("local", [])):
            conditions["local lesion"] = np.array(e["local"], int)
        for r, s in enumerate(e["random_damage_matched"][:CONFIG["N_RANDOM"]]):
            conditions[f"random {r + 1}"] = np.array(s, int)
    df, chk = run_model(name, site, conditions)
    obj_dfs.append(df)
    check_dfs.append(chk)
    rng = np.random.default_rng(C.derive_seed("stage4b_boot", name))
    err = (chk.DA_sampled - chk.DA_exact).abs()
    print(f"  sampling check ({len(chk)} objects, intact): per-object r = {np.corrcoef(chk.DA_sampled, chk.DA_exact)[0, 1]:0.3f} | "
          f"mean DA sampled {chk.DA_sampled.mean():0.3f} vs exact {chk.DA_exact.mean():0.3f} | "
          f"median |error| {err.median():0.3f} vs median |D| {chk.D.abs().median():0.3f}")
    print("    (per-object sampling error is part of the between-object spread, so the object bootstrap intervals include it)")
    piv = {k: d.set_index("source_id").loc[[o["source_id"] for o in OBJ]] for k, d in df.groupby("condition")}
    for k in conditions:
        if k.startswith("random"):
            continue
        d = piv[k]
        row = dict(model=name, condition=k)
        for comp in ["D", "DA", "DI"]:
            row[comp], row[f"{comp}_lo"], row[f"{comp}_hi"] = boot(d[comp].to_numpy(), rng, obj_cat)
        share = lambda z: z[:, 1].mean() / z[:, 0].mean() if abs(z[:, 0].mean()) > 1e-12 else np.nan
        row["share_DI"], row["share_DI_lo"], row["share_DI_hi"] = boot(d[["D", "DI"]].to_numpy(), rng, obj_cat, share)
        summ.append(row)
    rnd = [k for k in conditions if k.startswith("random")]
    if rnd:
        r_means = pd.DataFrame([{c: piv[k][c].mean() for c in ["D", "DA", "DI"]} for k in rnd])
        summ.append(dict(model=name, condition="random (damage-matched)",
                         **{c: r_means[c].mean() for c in ["D", "DA", "DI"]},
                         **{f"{c}_lo": r_means[c].quantile(0.025) for c in ["D", "DA", "DI"]},
                         **{f"{c}_hi": r_means[c].quantile(0.975) for c in ["D", "DA", "DI"]},
                         share_DI=float(r_means.DI.mean() / r_means.D.mean())))
        for k in ["configural lesion", "local lesion"]:
            if k not in piv:
                continue
            for comp in ["DA", "DI"]:
                ch = (piv[k][comp] - piv["intact"][comp]).to_numpy()
                m, lo, hi = boot(ch, rng, obj_cat)
                r_ch = np.array([(piv[r][comp] - piv["intact"][comp]).mean() for r in rnd])
                lesion_rows.append(dict(model=name, lesion=k, component=comp, change=m, lo=lo, hi=hi,
                                        random_mean_change=float(r_ch.mean()),
                                        random_min=float(r_ch.min()), random_max=float(r_ch.max()),
                                        frac_random_larger_drop=float((r_ch < m).mean()), n_random=len(r_ch)))

# =============================================================================
# 4. Save, tables, figure
# =============================================================================
obj_df = pd.concat(obj_dfs, ignore_index=True)
chk_df = pd.concat(check_dfs, ignore_index=True)
sum_df, les_df = pd.DataFrame(summ), pd.DataFrame(lesion_rows)
obj_df.to_csv(RES / f"object_components{sfx}.csv", index=False)
chk_df.to_csv(RES / f"sampling_check{sfx}.csv", index=False)
sum_df.to_csv(RES / f"components_summary{sfx}.csv", index=False)
les_df.to_csv(RES / f"lesion_effects{sfx}.csv", index=False)
C.save_json(dict(sets=SETS_PATH.name, config=CONFIG, n_objects=len(OBJ)), RES / f"stage4b_summary{sfx}.json")

fmt = lambda m, lo, hi: (f"{m:0.2f} [{lo:0.2f}, {hi:0.2f}]" if np.isfinite(lo) else f"{m:0.2f}") if np.isfinite(m) else "n/a"
lines = [r"\begin{tabular}{lllll}", r"\hline",
         r"Model & Lesion & Component & Change [95\% CI] & Random lesions: mean [min, max] \\", r"\hline"]
for r in les_df.itertuples():
    lines.append(f"{C.MODEL_INFO[r.model][0]} & {r.lesion} & {'position-coded parts' if r.component == 'DA' else 'interactions'} & "
                 f"{fmt(r.change, r.lo, r.hi)} & {fmt(r.random_mean_change, r.random_min, r.random_max)} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage4b_table.tex").write_text("\n".join(lines))


def nice(lo, hi):
    span = max(hi - lo, 1e-9)
    step = next(s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20) if span / s <= 6)
    return np.floor(lo / step) * step, np.ceil(hi / step) * step, step


def fig_components():
    """Configural effect on evidence split into position-coded parts (DA) and interactions (DI), by condition."""
    C.set_pub_style()
    conds = [("intact", "Intact"), ("configural lesion", "Configural\nlesion"), ("local lesion", "Local\nlesion"),
             ("random (damage-matched)", "Random\nlesions")]
    comps = [("D", "Total", "o", "#000000"), ("DA", "Position-coded parts", "s", "#2a78d6"),
             ("DI", "Interactions", "D", "#eb6834")]
    fig, axes = plt.subplots(1, len(models), figsize=(8.0 * len(models), 6.8),
                             gridspec_kw=dict(width_ratios=[4] * (len(models) - 1) + [1.6]))
    for p, (ax, (m, site)) in enumerate(zip(axes, models)):
        d = sum_df[sum_df.model == m]
        xs, xl, vals = [], [], [0.0]
        for i, (k, lab) in enumerate([c for c in conds if c[0] in set(d.condition)]):
            r = d[d.condition == k].iloc[0]
            for j, (c, cl, mk, col) in enumerate(comps):
                x = i + (j - 1) * 0.22
                ax.errorbar(x, r[c], yerr=[[r[c] - r[f"{c}_lo"]], [r[f"{c}_hi"] - r[c]]], fmt=mk, ms=9, capsize=4,
                            lw=1.8, color=col, mfc=col if c != "D" else "white", label=cl if (p == 0 and i == 0) else None)
                vals += [r[f"{c}_lo"], r[f"{c}_hi"]]
            xs.append(i)
            xl.append(lab)
        lo, hi, step = nice(min(vals), max(vals))
        C.style_axes(ax, ylim=(lo, hi), ystep=step, decimals=2 if step < 0.1 else (1 if step < 1 else 0))
        ax.axhline(0, color="#000000", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(xl)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_ylabel(f"Grid minus scrambled evidence\n{C.MODEL_INFO[m][0]}")
        ax.yaxis.set_label_coords(-0.13 if p < len(models) - 1 else -0.42, 0.5)
        ax.text(-0.13 if p < len(models) - 1 else -0.42, 1.03, "abcd"[p], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.45, bottom=0.25)
    C.save_figure(fig, FIGS / f"stage4b_components{sfx}")
    plt.close(fig)


fig_components()

pd.set_option("display.width", 240)
show = lambda d: d.round(3).astype(object).where(d.notna(), "").to_string(index=False)
print("\nConfigural effect on evidence and its components (objects; random = mean and 95% range over sets)")
print(show(sum_df))
print("\nLesion effects on each component (change from intact; random lesions for comparison)")
print(show(les_df))
print("\nDone. Outputs in", RES, "and", FIGS)
