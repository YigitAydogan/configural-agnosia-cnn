"""
STAGE 1: Stimulus battery and validation (v2)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run. The shared module used by every
stage (agnosia_common.py) is embedded below and written to DRIVE_ROOT on each run, so
later stages import the same code. Drive mounting is skipped if it is already mounted.

Changes from v1 (after the first run)
  - Binary silhouettes leave local-only models at chance (BagNet-33 control accuracy
    0.054 to 0.068 against chance 0.0625), so a behavioural local-equivalence test is
    uninformative on silhouettes. v2 adds a TEXTURED family: the same COCO instances with
    their original pixels inside the mask on a white background. Every manipulation is
    applied identically (same cuts, same permutations) to the silhouette and texture
    versions, so the two families are exactly aligned.
  - The behavioural gate is now evaluated on the texture family, where BagNet can
    recognise the controls. For silhouettes, local equivalence is established analytically
    (identical area, perimeter and mirror-pooled 3 x 3 patch statistics) and the BagNet
    result is reported as uninformative when it sits at floor.
  - Decisions with failed floor checks are reported as 'floor', not 'equivalent'.
  - Categories short of the COCO target (knife, bottle) get a second selection pass with a
    relaxed area threshold; the minimum bbox side (resolution) is unchanged. Category-
    balanced means are reported alongside source-weighted means.
  - Headroom diagnostics: whether ResNet-50's drop excludes zero, and the cost of the cuts
    themselves (the grid gaps cost ResNet-50 about 0.21 on silhouettes).

What this stage produces
  1. Sources: Geirhos silhouettes (160) and COCO 2017 instances (target 100 per category),
     each assigned to a 'localizer' or 'test' split (stratified 50/50)
  2. Configural battery, rendered as silhouette (battery 'configural', PNG) and, for COCO,
     as texture (battery 'configural_tex', JPEG q95, identical compression across conditions)
       intact, cutH_intact (control), frank_top, frank_bottom,
       cutG_intact (control), scrambled (5 variants silhouette, 3 texture)
  3. Navon battery, overlapping figures, model-vs-human reference sets
  4. Validation: low-level statistics, BagNet-9/17/33 and ResNet-50 predictions,
     equivalence gate, sensitivity analysis (grid cells >= 48 px)
"""

# =============================================================================
# 0. Setup
# =============================================================================
import os
import sys
import gc
import json
import time
import subprocess
from pathlib import Path

IN_COLAB = os.path.exists("/content") and "AGNOSIA_TEST" not in os.environ
if IN_COLAB:
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/wielandbrendel/bag-of-local-features-models.git",
                    "pycocotools"], check=True)

# -----------------------------------------------------------------------------
# Shared module (embedded). Written to DRIVE_ROOT/agnosia_common.py and imported.
# -----------------------------------------------------------------------------
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

import importlib
COMMON_DIR = Path(os.environ.get("AGNOSIA_ROOT", "/content/drive/MyDrive/AGNOSIA_CNN"))
COMMON_DIR.mkdir(parents=True, exist_ok=True)
(COMMON_DIR / "agnosia_common.py").write_text(AGNOSIA_COMMON_SRC)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
import agnosia_common as C
C = importlib.reload(C)   # pick up the freshly written version when the cell is re-run

import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt

STAGE = "stage1"
RUN_COCO = os.environ.get("AGNOSIA_NO_COCO") is None
RUN_MODELS = os.environ.get("AGNOSIA_NO_MODELS") is None
FAKE_TEX = os.environ.get("AGNOSIA_FAKE_TEX") is not None   # offline testing only

CONFIG = dict(
    VERSION=2,
    # COCO silhouette selection
    N_COCO_PER_CAT=100,
    COCO_SPLITS=["val2017", "train2017"],
    COCO_MIN_AREA_FRAC=0.05,          # instance area / image area, first pass
    COCO_MIN_AREA_FRAC_FALLBACK=0.01, # second pass, only for categories still short
    COCO_MIN_SIDE=96,                 # px, longer bbox side in the original image (resolution)
    COCO_BORDER=4,
    COCO_MAX_OCCLUSION=0.10,
    COCO_MIN_LCC=0.97,
    COCO_MIN_FILL=0.15,
    TARGET_SIDE=176,
    # Configural manipulations
    GAP=6,
    GRID=3,
    N_SCRAMBLE=5,
    N_SCRAMBLE_TEX=3,
    MIN_PIECE_PX=40,
    MIN_CELL_PX=48,
    MIN_MANIP_STRENGTH=0.10,
    JPEG_QUALITY=95,
    # Navon
    N_NAVON=10,
    NAVON_LETTERS=["E", "H", "L", "O", "T", "X"],
    # Overlapping figures
    N_OVERLAP_PAIRS=320,
    OVERLAP_LINE_PX=2,
    # Splits
    LOCALIZER_FRAC=0.5,
    # Validation
    MODELS=["bagnet9", "bagnet17", "bagnet33", "resnet50"],
    PRIMARY_LOCAL="bagnet33",
    PRIMARY_BATTERY="configural_tex",
    N_BOOT=10000,
    EQ_BOUND=0.05,
    FLOOR_MULT=2.0,
    BATCH=128,
)

FORCE_STIMULI = False
FORCE_PREDICTIONS = False

C.ensure_dirs()
C.set_seed()
STIM_ROOT = C.STIM_LOCAL / STAGE
RES = C.RESULTS_DIR / STAGE
FIGS = C.FIG_DIR / STAGE
for d in [STIM_ROOT, RES, FIGS, FIGS / "qc"]:
    d.mkdir(parents=True, exist_ok=True)

MANIFEST_DRIVE = C.STIM_DRIVE / f"{STAGE}_manifest.csv"
CONFIG_DRIVE = C.STIM_DRIVE / f"{STAGE}_config.json"
QC_EXCLUSIONS = C.STIM_DRIVE / f"{STAGE}_qc_exclusions.json"   # optional, list of source_ids
COCO_IMG_LOCAL = C.RAW_LOCAL / "coco" / "images"
COCO_IMG_DRIVE = C.RAW_DRIVE / "coco" / "images_selected.tar"
CFG_HASH = C.config_hash({k: v for k, v in CONFIG.items()
                          if k not in ("MODELS", "PRIMARY_LOCAL", "PRIMARY_BATTERY", "N_BOOT",
                                       "EQ_BOUND", "FLOOR_MULT", "BATCH", "MIN_CELL_PX")}
                         | dict(fake_tex=FAKE_TEX, coco=RUN_COCO))
H = W = C.IMG_SIZE
HALF = CONFIG["GAP"] // 2
BATTERIES = ["configural", "configural_tex"]
BATTERY_LABEL = {"configural": "Silhouette", "configural_tex": "Texture"}

print(f"Stage 1 v2 | config hash {CFG_HASH} | COCO {RUN_COCO} | models {RUN_MODELS}")


# =============================================================================
# 1. Mask and layer utilities
#    A 'layer' array L has shape (H, W, 4): channel 0 is the object mask (0/1),
#    channels 1-3 are RGB with white background. All manipulations act on layers,
#    so silhouette and texture renderings are produced by the same geometric operation.
# =============================================================================
def to_png(mask, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def to_jpg(L, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(L[..., 1:])).save(path, quality=CONFIG["JPEG_QUALITY"],
                                                            subsampling=0)


def bbox(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def resize_mask(mask, out_h, out_w):
    img = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(img.resize((out_w, out_h), Image.BILINEAR)) > 127


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
    """Resize crop so its longer side = side, paste (OR) centred at (cy, cx)."""
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
# 2. Sources
# =============================================================================
def load_geirhos_sources():
    d = C.get_mvh_dataset("silhouette")
    rows = []
    for cat in C.CATEGORIES:
        for p in sorted((d / cat).glob("*.png")):
            a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            mask = a < 128
            src = dict(source_set="geirhos", source_id=f"geirhos_{p.stem}", category=cat,
                       mask=mask, tex=None, origin=str(p.name))
            if FAKE_TEX:
                rng = np.random.default_rng(C.derive_seed("faketex", p.stem))
                noise = ndimage.gaussian_filter(rng.random((H, W, 3)) * 255, sigma=(3, 3, 0))
                src["tex"] = bg_layers()
                src["tex"][mask, 0] = 1
                src["tex"][mask, 1:] = np.clip(noise, 0, 255).astype(np.uint8)[mask]
            rows.append(src)
    print(f"  Geirhos silhouettes: {len(rows)}")
    return rows


def select_coco(excluded):
    """Select instances; returns per-category lists of records holding the full-resolution
    crop mask (packed) and the image file needed for the texture rendering."""
    from pycocotools.coco import COCO
    ann_dir = C.get_coco_annotations()
    n_target = CONFIG["N_COCO_PER_CAT"]
    selected = {c: [] for c in C.CATEGORIES}
    used_images = {c: set() for c in C.CATEGORIES}
    taken = set()
    reject_log = {c: dict(considered=0, small=0, border=0, occluded=0, fragmented=0, sparse=0,
                          excluded_qc=0, duplicate_image=0) for c in C.CATEGORIES}
    thresholds = [CONFIG["COCO_MIN_AREA_FRAC"], CONFIG["COCO_MIN_AREA_FRAC_FALLBACK"]]
    for thr in thresholds:
        for split in CONFIG["COCO_SPLITS"]:
            if all(len(v) >= n_target for v in selected.values()):
                break
            t0 = time.time()
            coco = COCO(str(ann_dir / f"instances_{split}.json"))
            print(f"  loaded COCO {split} in {time.time() - t0:0.0f} s (area threshold {thr})")
            for cat in C.CATEGORIES:
                if len(selected[cat]) >= n_target:
                    continue
                cat_id = coco.getCatIds(catNms=[cat])[0]
                anns = coco.loadAnns(coco.getAnnIds(catIds=[cat_id], iscrowd=False))
                order = np.random.default_rng(C.derive_seed("coco", split, cat)).permutation(len(anns))
                log = reject_log[cat]
                for i in order:
                    if len(selected[cat]) >= n_target:
                        break
                    ann = anns[i]
                    sid = f"coco_{ann['id']}"
                    if sid in taken:
                        continue
                    log["considered"] += 1
                    if sid in excluded:
                        log["excluded_qc"] += 1
                        continue
                    if ann["image_id"] in used_images[cat]:
                        log["duplicate_image"] += 1
                        continue
                    img = coco.imgs[ann["image_id"]]
                    ih, iw = img["height"], img["width"]
                    x, y, w, h = ann["bbox"]
                    if ann["area"] < thr * ih * iw or max(w, h) < CONFIG["COCO_MIN_SIDE"]:
                        log["small"] += 1
                        continue
                    b = CONFIG["COCO_BORDER"]
                    if x <= b or y <= b or x + w >= iw - b or y + h >= ih - b:
                        log["border"] += 1
                        continue
                    occ = 0.0
                    for other in coco.imgToAnns[ann["image_id"]]:
                        if other["id"] == ann["id"]:
                            continue
                        ox, oy, ow, oh = other["bbox"]
                        iw_ = max(0.0, min(x + w, ox + ow) - max(x, ox))
                        ih_ = max(0.0, min(y + h, oy + oh) - max(y, oy))
                        occ = max(occ, iw_ * ih_ / (w * h))
                    if occ > CONFIG["COCO_MAX_OCCLUSION"]:
                        log["occluded"] += 1
                        continue
                    m = coco.annToMask(ann).astype(bool)
                    lab, n = ndimage.label(m)
                    if n == 0:
                        log["fragmented"] += 1
                        continue
                    sizes = ndimage.sum(m, lab, range(1, n + 1))
                    if sizes.max() / sizes.sum() < CONFIG["COCO_MIN_LCC"]:
                        log["fragmented"] += 1
                        continue
                    m = lab == (np.argmax(sizes) + 1)
                    r0, r1, c0, c1 = bbox(m)
                    crop = m[r0:r1, c0:c1]
                    if crop.mean() < CONFIG["COCO_MIN_FILL"]:
                        log["sparse"] += 1
                        continue
                    selected[cat].append(dict(
                        source_id=sid, category=cat, split_coco=split, file_name=img["file_name"],
                        crop_bits=np.packbits(crop), crop_shape=crop.shape, img_bbox=(r0, r1, c0, c1),
                        area_threshold=thr, origin=f"{split}/img{ann['image_id']}/ann{ann['id']}"))
                    used_images[cat].add(ann["image_id"])
                    taken.add(sid)
            del coco
            gc.collect()
    counts = {c: len(selected[c]) for c in C.CATEGORIES}
    relaxed = {c: sum(r["area_threshold"] != thresholds[0] for r in selected[c]) for c in C.CATEGORIES}
    print(f"  COCO instances: {sum(counts.values())} | per category {counts}")
    print(f"  of which from relaxed area pass: { {c: n for c, n in relaxed.items() if n} }")
    short = [c for c, n in counts.items() if n < n_target]
    if short:
        print(f"  WARNING: categories below target after relaxed pass: {short}")
    C.save_json(dict(counts=counts, relaxed_pass=relaxed, rejections=reject_log),
                RES / "coco_selection_log.json")
    return selected


def fetch_coco_images(records):
    """Download the selected images in parallel; cache the set on Drive as one tar."""
    COCO_IMG_LOCAL.mkdir(parents=True, exist_ok=True)
    if COCO_IMG_DRIVE.exists() and not any(COCO_IMG_LOCAL.rglob("*.jpg")):
        print("  extracting cached COCO images from Drive")
        with tarfile.open(COCO_IMG_DRIVE) as t:
            t.extractall(COCO_IMG_LOCAL)
    todo = [(r["split_coco"], r["file_name"]) for r in records
            if not (COCO_IMG_LOCAL / r["split_coco"] / r["file_name"]).exists()]
    todo = sorted(set(todo))

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
        print(f"  downloading {len(todo)} COCO images")
        t0 = time.time()
        with ThreadPoolExecutor(32) as ex:
            ok = list(ex.map(get, todo))
        print(f"  downloaded {sum(ok)} / {len(todo)} in {time.time() - t0:0.0f} s")
        if IN_COLAB:
            tmp = Path("/tmp/images_selected.tar")
            with tarfile.open(tmp, "w") as t:
                for sp in sorted(p.name for p in COCO_IMG_LOCAL.iterdir() if p.is_dir()):
                    t.add(COCO_IMG_LOCAL / sp, arcname=sp)
            COCO_IMG_DRIVE.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(tmp, COCO_IMG_DRIVE)
            tmp.unlink()


def load_coco_sources(excluded):
    selected = select_coco(excluded)
    records = [r for c in C.CATEGORIES for r in selected[c]]
    fetch_coco_images(records)
    rows, n_missing = [], 0
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
                    tex = place_layers(crop, rgb, CONFIG["TARGET_SIDE"], H / 2, W / 2)
            except Exception:
                tex = None
        if tex is not None:
            mask = mask_of(tex)          # identical geometry to the texture rendering
        else:
            n_missing += 1
            mask = place(np.zeros((H, W), bool), crop, CONFIG["TARGET_SIDE"], H / 2, W / 2)
        rows.append(dict(source_set="coco", source_id=r["source_id"], category=r["category"],
                         mask=mask, tex=tex, origin=r["origin"], area_threshold=r["area_threshold"]))
    print(f"  COCO sources: {len(rows)} | texture unavailable for {n_missing}")
    return rows


def assign_splits(sources):
    df = pd.DataFrame([{k: v for k, v in s.items() if k in ("source_set", "source_id", "category")}
                       for s in sources])
    split = {}
    for (sset, cat), g in df.groupby(["source_set", "category"]):
        ids = g["source_id"].to_numpy()
        perm = np.random.default_rng(C.derive_seed("split", sset, cat)).permutation(len(ids))
        n_loc = int(round(len(ids) * CONFIG["LOCALIZER_FRAC"]))
        for j, k in enumerate(perm):
            split[ids[k]] = "localizer" if j < n_loc else "test"
    for s in sources:
        s["split"] = split[s["source_id"]]
    return sources


def qc_contact_sheets(sources):
    """One sheet per category for manual inspection of COCO sources (texture shown when available).
    To exclude bad ones, list their source_ids in stage1_qc_exclusions.json on Drive and re-run
    with FORCE_STIMULI = True; the selector backfills with the next eligible instances."""
    coco = [s for s in sources if s["source_set"] == "coco"]
    for cat in C.CATEGORIES:
        items = [s for s in coco if s["category"] == cat]
        if not items:
            continue
        ncol = 10
        nrow = int(np.ceil(len(items) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.8))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, s in zip(axes.ravel(), items):
            if s["tex"] is not None:
                ax.imshow(s["tex"][..., 1:])
            else:
                ax.imshow(~s["mask"], cmap="gray", vmin=0, vmax=1)
            ax.text(0.5, -0.06, s["source_id"].replace("coco_", ""), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7)
        fig.savefig(FIGS / "qc" / f"coco_{cat}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 3. Configural manipulations (on layers)
# =============================================================================
def hsplit_control(L):
    r0, r1, _, _ = bbox(mask_of(L))
    ym = (r0 + r1) // 2
    ctrl = L.copy()
    clear(ctrl[max(0, ym - HALF):ym + HALF, :])
    return ctrl, ym


def seam_centre(part_mask, row, step, fallback):
    for k in range(12):
        r = row + k * step
        if 0 <= r < part_mask.shape[0]:
            cols = np.where(part_mask[r])[0]
            if cols.size:
                return cols.mean()
    return fallback


def frankenstein(L, which):
    ctrl, ym = hsplit_control(L)
    top = ctrl.copy(); clear(top[ym:, :])
    bot = ctrl.copy(); clear(bot[:ym, :])
    mt, mb = mask_of(top), mask_of(bot)
    if mt.sum() == 0 or mb.sum() == 0:
        return None
    cent = lambda m: np.where(m)[1].mean()
    if which == "top":
        part, other, mp, mo = top, bot, mt, mb
        cp = seam_centre(mp, ym - HALF - 1, -1, cent(mp))
        co = seam_centre(mo, ym + HALF, +1, cent(mo))
    else:
        part, other, mp, mo = bot, top, mb, mt
        cp = seam_centre(mp, ym + HALF, +1, cent(mp))
        co = seam_centre(mo, ym - HALF - 1, -1, cent(mo))
    flipped = part[:, ::-1]
    dx_ideal = int(round(co - (W - 1 - cp)))
    cols = np.where(mask_of(flipped).any(axis=0))[0]
    dx = int(np.clip(dx_ideal, -cols[0], W - 1 - cols[-1]))
    moved = shift_cols_layers(flipped, dx)
    out = paste_over(other.copy(), moved)
    lost = 1.0 - mask_of(moved).sum() / max(1, mp.sum())
    return dict(control=ctrl, manip=out, area_lost_frac=float(lost), seam_offset_px=int(abs(dx_ideal - dx)))


def grid_geometry(mask):
    G = CONFIG["GRID"]
    r0, r1, c0, c1 = bbox(mask)
    h, w = r1 - r0, c1 - c0
    ch, cw = min(int(np.ceil(h / G)), H // G), min(int(np.ceil(w / G)), W // G)
    R0 = int(np.clip(r0 - (G * ch - h) // 2, 0, H - G * ch))
    C0 = int(np.clip(c0 - (G * cw - w) // 2, 0, W - G * cw))
    return R0, C0, ch, cw


def grid_control(L):
    """Zero a gap of HALF px on every side of every cell, so pieces never touch after permutation."""
    G = CONFIG["GRID"]
    R0, C0, ch, cw = grid_geometry(mask_of(L))
    ctrl = bg_layers()   # content outside the grid (objects touching the canvas edge) is dropped
    ctrl[R0:R0 + G * ch, C0:C0 + G * cw] = L[R0:R0 + G * ch, C0:C0 + G * cw]
    for i in range(G):
        for j in range(G):
            cell = ctrl[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw]
            clear(cell[:HALF]); clear(cell[-HALF:]); clear(cell[:, :HALF]); clear(cell[:, -HALF:])
    return ctrl, (R0, C0, ch, cw)


def grid_permutation(ctrl, geom, seed):
    """Derangement of non-empty cells maximising (up to MIN_MANIP_STRENGTH) the change in layout."""
    G = CONFIG["GRID"]
    R0, C0, ch, cw = geom
    cm = mask_of(ctrl)
    cells = [cm[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw] for i in range(G) for j in range(G)]
    nonempty = [k for k, c in enumerate(cells) if c.sum() >= CONFIG["MIN_PIECE_PX"]]
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
            best = (perm, strength)
        if strength >= CONFIG["MIN_MANIP_STRENGTH"]:
            break
    if best is None:
        return None
    return best[0], best[1], len(nonempty)


def apply_permutation(ctrl, geom, perm):
    G = CONFIG["GRID"]
    R0, C0, ch, cw = geom
    cells = [ctrl[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw].copy()
             for i in range(G) for j in range(G)]
    out = bg_layers()
    for k, cell in enumerate(cells):
        i, j = divmod(perm[k], G)
        paste_over(out[R0 + i * ch:R0 + (i + 1) * ch, C0 + j * cw:C0 + (j + 1) * cw], cell)
    return out


# =============================================================================
# 4. Navon battery
# =============================================================================
NAVON_BITMAPS = {
    "E": ["11111", "10000", "11110", "10000", "11111"],
    "H": ["10001", "10001", "11111", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100"],
    "X": ["10001", "01010", "00100", "01010", "10001"],
}
NAVON_CELLS = {L: [(r, c) for r, row in enumerate(b) for c, ch in enumerate(row) if ch == "1"]
               for L, b in NAVON_BITMAPS.items()}


def navon_font(size):
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(path), size)


def render_navon(cells, local, rng, square=False):
    spacing = int(rng.integers(32, 39))
    size = int(rng.integers(22, 29))
    oy, ox = rng.integers(-8, 9, size=2)
    img = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(img)
    font = navon_font(size)
    y_start = H / 2 - 2 * spacing + oy
    x_start = W / 2 - 2 * spacing + ox
    for r, c in cells:
        cy, cx = y_start + r * spacing, x_start + c * spacing
        if square:
            s = 0.3 * size
            dr.rectangle([cx - s, cy - s, cx + s, cy + s], fill=0)
        else:
            dr.text((cx, cy), local, fill=0, font=font, anchor="mm")
    return img, dict(spacing=spacing, font_px=size)


def build_navon(rows):
    letters = CONFIG["NAVON_LETTERS"]
    n = CONFIG["N_NAVON"]

    def save(img, rel, **meta):
        (STIM_ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(STIM_ROOT / rel)
        rows.append(dict(path=f"{STAGE}/{rel}", battery="navon", source_set="navon", category="",
                         split="all", **meta))

    for g in letters:
        for l in letters:
            cond = "navon_congruent" if g == l else "navon_incongruent"
            for v in range(n):
                rng = np.random.default_rng(C.derive_seed("navon", g, l, v))
                img, meta = render_navon(NAVON_CELLS[g], l, rng)
                save(img, Path("navon") / cond / f"G{g}_L{l}__v{v}.png", condition=cond,
                     source_id=f"navon_G{g}_L{l}", variant=v, global_letter=g, local_letter=l,
                     n_elements=len(NAVON_CELLS[g]), **meta)
        for v in range(n):
            rng = np.random.default_rng(C.derive_seed("navon_global_only", g, v))
            img, meta = render_navon(NAVON_CELLS[g], "", rng, square=True)
            save(img, Path("navon") / "navon_global_only" / f"G{g}__v{v}.png", condition="navon_global_only",
                 source_id=f"navon_G{g}", variant=v, global_letter=g, local_letter="",
                 n_elements=len(NAVON_CELLS[g]), **meta)
    counts = [len(NAVON_CELLS[g]) for g in letters]
    all_cells = [(r, c) for r in range(5) for c in range(5)]
    for l in letters:
        for v in range(n):
            rng = np.random.default_rng(C.derive_seed("navon_local_only", l, v))
            k = counts[v % len(counts)]
            cells = [all_cells[i] for i in rng.choice(25, size=k, replace=False)]
            img, meta = render_navon(cells, l, rng)
            save(img, Path("navon") / "navon_local_only" / f"L{l}__v{v}.png", condition="navon_local_only",
                 source_id=f"navon_L{l}", variant=v, global_letter="", local_letter=l, n_elements=k, **meta)


# =============================================================================
# 5. Overlapping figures
# =============================================================================
def build_overlap(sources, rows):
    px = CONFIG["OVERLAP_LINE_PX"]
    crops = {s["source_id"]: crop_to_bbox(s["mask"]) for s in sources}
    meta = {s["source_id"]: s for s in sources}
    for s in sources:
        canvas = place(np.zeros((H, W), bool), crops[s["source_id"]], 170, H / 2, W / 2)
        rel = Path("overlap") / "outline_single" / f"{s['source_id']}.png"
        to_png(outline(canvas, px), STIM_ROOT / rel)
        rows.append(dict(path=f"{STAGE}/{rel}", battery="overlap", condition="outline_single",
                         source_set=s["source_set"], source_id=s["source_id"], category=s["category"],
                         split=s["split"], variant=0))
    rng = np.random.default_rng(C.derive_seed("overlap_pairs"))
    by_split = {sp: [s["source_id"] for s in sources if s["split"] == sp] for sp in ("localizer", "test")}
    n_per_split = CONFIG["N_OVERLAP_PAIRS"] // 2
    for sp, ids in by_split.items():
        made, guard = 0, 0
        seen = set()
        while made < n_per_split and guard < 100000:
            guard += 1
            a, b = rng.choice(ids, size=2, replace=False)
            if meta[a]["category"] == meta[b]["category"] or (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            pid = f"pair{sp[:3]}{made:04d}"
            j = rng.integers(-10, 11, size=4)
            variants = {
                "overlap_large": [(170, H / 2 + j[0], W / 2 + j[1]), (170, H / 2 + j[2], W / 2 + j[3])],
                "overlap_small": [(100, H / 2, W / 2 - 25), (100, H / 2, W / 2 + 25)],
                "side_by_side":  [(100, H / 2, 58), (100, H / 2, 166)],
            }
            for cond, specs in variants.items():
                ink = np.zeros((H, W), bool)
                single_ink = 0
                for sid, (side, cy, cx) in zip((a, b), specs):
                    o = outline(place(np.zeros((H, W), bool), crops[sid], side, cy, cx), px)
                    single_ink += o.sum()
                    ink |= o
                rel = Path("overlap") / cond / f"{pid}.png"
                to_png(ink, STIM_ROOT / rel)
                rows.append(dict(path=f"{STAGE}/{rel}", battery="overlap", condition=cond,
                                 source_set="pair", source_id=pid, category=meta[a]["category"],
                                 category_b=meta[b]["category"], source_a=a, source_b=b, split=sp,
                                 variant=0, ink_px=int(ink.sum()),
                                 ink_overlap_frac=float(1 - ink.sum() / max(1, single_ink))))
            made += 1


# =============================================================================
# 6. Reference sets
# =============================================================================
def build_reference(rows):
    import shutil
    for name in ["silhouette", "edge", "sketch", "cue-conflict"]:
        d = C.get_mvh_dataset(name)
        dst = STIM_ROOT / "reference" / name
        if not dst.exists():
            shutil.copytree(d, dst)
        for p in sorted(dst.rglob("*.png")):
            rel = p.relative_to(STIM_ROOT)
            row = dict(path=f"{STAGE}/{rel}", battery="reference", condition=f"ref_{name}",
                       source_set="model-vs-human", source_id=f"{name}_{p.stem}", split="reference",
                       variant=0)
            if name in ("silhouette", "edge"):
                row["category"] = p.parent.name
            elif name == "sketch":
                row["category"] = p.name.split("_")[4]
            else:
                shape_cat = p.parent.name
                tex = p.stem.split("-")[-1].rstrip("0123456789")
                row.update(category=shape_cat, texture_category=tex, cue_conflict_same=(shape_cat == tex))
            rows.append(row)


# =============================================================================
# 7. Build everything
# =============================================================================
def build_configural(sources, rows):
    n_skip = dict(frank=0, scramble=0)
    for s in sources:
        sid = s["source_id"]
        L_sil = layers_from_mask(s["mask"])
        renders = [("configural", L_sil, CONFIG["N_SCRAMBLE"])]
        if s["tex"] is not None:
            renders.append(("configural_tex", s["tex"], CONFIG["N_SCRAMBLE_TEX"]))

        # geometry is decided once (on the silhouette) and applied to every rendering
        ctrlG_sil, geom = grid_control(L_sil)
        cell_px = int(min(geom[2], geom[3]))
        perms = []
        for v in range(CONFIG["N_SCRAMBLE"]):
            r = grid_permutation(ctrlG_sil, geom, C.derive_seed("scramble", sid, v))
            perms.append(r)
            if r is None:
                n_skip["scramble"] += 1

        for battery, L, n_scr in renders:
            base = dict(battery=battery, source_set=s["source_set"], source_id=sid,
                        category=s["category"], split=s["split"],
                        area_threshold=s.get("area_threshold", np.nan))

            def emit(cond, LL, variant=0, **extra):
                if battery == "configural":
                    rel = Path("configural") / cond / f"{sid}__v{variant}.png"
                    to_png(mask_of(LL), STIM_ROOT / rel)
                else:
                    rel = Path("configural_tex") / cond / f"{sid}__v{variant}.jpg"
                    to_jpg(LL, STIM_ROOT / rel)
                rows.append(dict(path=f"{STAGE}/{rel}", condition=cond, variant=variant, **base, **extra))

            emit("intact", L)
            ctrlH, _ = hsplit_control(L)
            emit("cutH_intact", ctrlH)
            for which in ("top", "bottom"):
                f = frankenstein(L, which)
                if f is None:
                    if battery == "configural":
                        n_skip["frank"] += 1
                    continue
                emit(f"frank_{which}", f["manip"], area_lost_frac=f["area_lost_frac"],
                     seam_offset_px=f["seam_offset_px"],
                     manip_strength=1 - iou(mask_of(f["control"]), mask_of(f["manip"])))
            ctrlG, _ = grid_control(L)
            emit("cutG_intact", ctrlG, cell_px=cell_px)
            for v in range(n_scr):
                if perms[v] is None:
                    continue
                perm, strength, n_pieces = perms[v]
                emit("scrambled", apply_permutation(ctrlG, geom, perm), variant=v, cell_px=cell_px,
                     n_pieces=n_pieces, perm=" ".join(map(str, perm)), manip_strength=strength)
    print(f"  configural images: {len(rows)} | skipped {n_skip}")


def build_stimuli():
    excluded = set(C.load_json(QC_EXCLUSIONS)) if QC_EXCLUSIONS.exists() else set()
    if excluded:
        print(f"  QC exclusions: {len(excluded)}")
    import shutil
    if STIM_ROOT.exists():
        shutil.rmtree(STIM_ROOT)
    STIM_ROOT.mkdir(parents=True)
    sources = load_geirhos_sources()
    if RUN_COCO:
        sources += load_coco_sources(excluded)
    sources = [s for s in sources if s["source_id"] not in excluded]
    sources = assign_splits(sources)
    qc_contact_sheets(sources)

    for sset in sorted({s["source_set"] for s in sources}):
        ss = [s for s in sources if s["source_set"] == sset]
        np.savez_compressed(STIM_ROOT / f"masks_{sset}.npz",
                            source_ids=np.array([s["source_id"] for s in ss]),
                            masks=np.stack([s["mask"] for s in ss]))

    rows = []
    build_configural(sources, rows)
    build_navon(rows)
    build_overlap(sources, rows)
    build_reference(rows)

    man = pd.DataFrame(rows)
    man.to_csv(STIM_ROOT / "manifest.csv", index=False)
    man.to_csv(MANIFEST_DRIVE, index=False)
    C.save_json(dict(config=CONFIG, hash=CFG_HASH, n_images=len(man),
                     created=time.strftime("%Y-%m-%d %H:%M:%S")), CONFIG_DRIVE)
    print(f"  manifest: {len(man)} images")
    if IN_COLAB:
        print("  archiving stimuli to Drive ...")
        C.pack_stimuli(STAGE)
    return man


def get_stimuli():
    if not FORCE_STIMULI and CONFIG_DRIVE.exists() and MANIFEST_DRIVE.exists():
        cfg = C.load_json(CONFIG_DRIVE)
        if cfg.get("hash") == CFG_HASH and (C.stimuli_archive_path(STAGE).exists() or not IN_COLAB):
            print("  reusing cached stimuli")
            if IN_COLAB:
                C.unpack_stimuli(STAGE)
            return pd.read_csv(MANIFEST_DRIVE, low_memory=False)
    return build_stimuli()


print("\n[1] Stimuli")
man = get_stimuli()
print(man.groupby(["battery", "condition"]).size().to_string())


# =============================================================================
# 8. Low-level statistics
# =============================================================================
def load_mask(rel):
    return np.asarray(Image.open(C.STIM_LOCAL / rel).convert("L")) < 128


def load_rgb(rel):
    return np.asarray(Image.open(C.STIM_LOCAL / rel).convert("RGB"), dtype=np.float64)


_POW = (2 ** np.arange(9)).reshape(3, 3)


def _build_mirror_map():
    # correlate places weight 2 ** (3i + j) on neighbour (i, j); a horizontal mirror maps (i, j) -> (i, 2 - j)
    mp = np.zeros(512, dtype=np.int64)
    for c in range(512):
        out = 0
        for k in range(9):
            i, j = divmod(k, 3)
            out |= ((c >> k) & 1) << (3 * i + (2 - j))
        mp[c] = out
    return mp


MIRROR_MAP = _build_mirror_map()


def patch_hist(mask):
    m = mask.astype(np.int64)
    codes = ndimage.correlate(m, _POW, mode="constant", cval=0)[1:-1, 1:-1].ravel()
    codes = codes[(codes != 0) & (codes != 511)]
    canon = np.minimum(codes, MIRROR_MAP[codes])
    h = np.bincount(canon, minlength=512).astype(np.float64)
    return h / max(1.0, h.sum())


def js_div(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / b[nz]))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def orient_hist(img, nb=6):
    f = ndimage.gaussian_filter(img.astype(np.float64), 1.0)
    gy, gx = np.gradient(f)
    mag = np.hypot(gx, gy)
    th = np.mod(np.arctan2(gy, gx), np.pi)
    th = np.minimum(th, np.pi - th)            # mirror-pooled
    h, _ = np.histogram(th, bins=nb, range=(0, np.pi / 2), weights=mag)
    return h / max(1e-12, h.sum())


def spectral_bands(img):
    a = img.astype(np.float64)
    a = a - a.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    tot = P.sum()
    if tot <= 0:
        return dict(sf_low=np.nan, sf_mid=np.nan, sf_high=np.nan)
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - H / 2, xx - W / 2)          # cycles per image
    return dict(sf_low=float(P[r < 4].sum() / tot), sf_mid=float(P[(r >= 4) & (r < 16)].sum() / tot),
                sf_high=float(P[r >= 16].sum() / tot))


def colour_hist(rgb, bins=4):
    q = np.clip((rgb / 256 * bins).astype(int), 0, bins - 1)
    code = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
    h = np.bincount(code.ravel(), minlength=bins ** 3).astype(np.float64)
    return h / h.sum()


def lowlevel_sil(rel):
    mask = load_mask(rel)
    d = dict(area=int(mask.sum()), perimeter=int(outline(mask, 1).sum()),
             n_components=int(ndimage.label(mask)[1]))
    d.update(spectral_bands(mask))
    return d, orient_hist(mask), patch_hist(mask)


def lowlevel_tex(rel):
    rgb = load_rgb(rel)
    lum = rgb.mean(axis=2)
    d = dict(lum_mean=float(lum.mean()), lum_sd=float(lum.std()))
    d.update(spectral_bands(lum))
    return d, orient_hist(lum), colour_hist(rgb)


PAIRS = [("frankenstein", "cutH_intact", ["frank_top", "frank_bottom"]),
         ("scramble", "cutG_intact", ["scrambled"])]


def run_lowlevel(man):
    recs, ref = [], []
    for battery in BATTERIES:
        cfg = man[man.battery == battery]
        if cfg.empty:
            continue
        fn = lowlevel_sil if battery == "configural" else lowlevel_tex
        for fam, ctrl_cond, manip_conds in PAIRS:
            ctrl_rows = cfg[cfg.condition == ctrl_cond].set_index("source_id")
            cache = {sid: fn(r.path) for sid, r in ctrl_rows.iterrows()}
            for r in cfg[cfg.condition.isin(manip_conds)].itertuples():
                if r.source_id not in cache:
                    continue
                dc, oc, hc = cache[r.source_id]
                dm, om, hm = fn(r.path)
                rec = dict(battery=battery, family=fam, condition=r.condition, source_set=r.source_set,
                           source_id=r.source_id, category=r.category, variant=r.variant,
                           orient_L1=float(np.abs(om - oc).sum()),
                           d_sf_low=dm["sf_low"] - dc["sf_low"], d_sf_mid=dm["sf_mid"] - dc["sf_mid"],
                           d_sf_high=dm["sf_high"] - dc["sf_high"])
                if battery == "configural":
                    rec.update(d_area_rel=(dm["area"] - dc["area"]) / max(1, dc["area"]),
                               d_perimeter_rel=(dm["perimeter"] - dc["perimeter"]) / max(1, dc["perimeter"]),
                               d_components=dm["n_components"] - dc["n_components"],
                               hist_div=js_div(hm, hc))
                else:
                    rec.update(d_lum_mean=dm["lum_mean"] - dc["lum_mean"], d_lum_sd=dm["lum_sd"] - dc["lum_sd"],
                               hist_div=float(np.abs(hm - hc).sum()))
                recs.append(rec)
            # between-exemplar reference (same category and set, different source)
            for (sset, cat), g in ctrl_rows.reset_index().groupby(["source_set", "category"]):
                ids = g.source_id.tolist()
                if len(ids) < 2:
                    continue
                rng = np.random.default_rng(C.derive_seed("between", battery, fam, sset, cat))
                for k, sid in enumerate(ids):
                    other = ids[(k + 1 + int(rng.integers(0, len(ids) - 1))) % len(ids)]
                    (_, o1, h1), (_, o2, h2) = cache[sid], cache[other]
                    div = js_div(h1, h2) if battery == "configural" else float(np.abs(h1 - h2).sum())
                    ref.append(dict(battery=battery, family=fam, source_set=sset,
                                    orient_L1=float(np.abs(o1 - o2).sum()), hist_div=div))
    ll, ref = pd.DataFrame(recs), pd.DataFrame(ref)
    summ = []
    for (battery, fam, sset), g in ll.groupby(["battery", "family", "source_set"]):
        rg = ref[(ref.battery == battery) & (ref.family == fam) & (ref.source_set == sset)]
        row = dict(battery=battery, family=fam, source_set=sset, n_pairs=len(g))
        cols = ["orient_L1", "hist_div", "d_sf_low", "d_sf_mid", "d_sf_high"]
        cols += (["d_area_rel", "d_perimeter_rel", "d_components"] if battery == "configural"
                 else ["d_lum_mean", "d_lum_sd"])
        for col in cols:
            row[f"{col}_median"] = float(g[col].median())
            row[f"{col}_q95abs"] = float(g[col].abs().quantile(0.95))
        row["orient_ratio_to_between"] = row["orient_L1_median"] / max(1e-12, float(rg.orient_L1.median()))
        row["hist_ratio_to_between"] = row["hist_div_median"] / max(1e-12, float(rg.hist_div.median()))
        summ.append(row)
    summ = pd.DataFrame(summ)
    ll.to_csv(RES / "lowlevel_pairs.csv", index=False)
    summ.to_csv(RES / "lowlevel_summary.csv", index=False)
    man[man.battery == "navon"].groupby("condition")["n_elements"].describe().to_csv(RES / "navon_element_counts.csv")
    return ll, summ


def lowlevel_flags(summ):
    """hist_div: mirror-pooled 3 x 3 binary patch JS divergence (silhouette) or colour histogram L1 (texture)."""
    flags = []
    for r in summ.itertuples():
        tag = f"{r.battery}/{r.family}/{r.source_set}"
        if r.battery == "configural":
            if r.d_area_rel_q95abs > 0.02:
                flags.append(f"{tag}: area change > 2% in 5% of pairs")
            if r.d_components_q95abs > 0:
                flags.append(f"{tag}: component count changed in some pairs")
        else:
            if r.d_lum_mean_q95abs > 1.0:
                flags.append(f"{tag}: mean luminance change > 1 grey level in 5% of pairs")
        if r.hist_ratio_to_between > 0.25:
            flags.append(f"{tag}: local histogram divergence is {r.hist_ratio_to_between:0.2f} of between-exemplar")
    return flags


print("\n[2] Low-level statistics")
ll, ll_summary = run_lowlevel(man)
print(ll_summary[["battery", "family", "source_set", "n_pairs", "hist_ratio_to_between",
                  "orient_ratio_to_between", "d_sf_low_median"]].round(4).to_string(index=False))


# =============================================================================
# 9. Model predictions
# =============================================================================
def run_predictions(man):
    import torch
    device = C.get_device()
    eval_rows = man[man.battery.isin(BATTERIES) | (man.condition == "ref_silhouette")].reset_index(drop=True)
    paths = [str(C.STIM_LOCAL / p) for p in eval_rows.path]
    preds = {}
    for name in CONFIG["MODELS"]:
        out = RES / f"predictions_{name}.csv"
        if out.exists() and not FORCE_PREDICTIONS:
            df = pd.read_csv(out)
            if len(df) == len(eval_rows) and df.path.tolist() == eval_rows.path.tolist():
                preds[name] = df
                print(f"  {name}: cached")
                continue
        t0 = time.time()
        model = C.load_model(name, device)
        p16 = C.predict_probs16(model, paths, device, batch_size=CONFIG["BATCH"])
        del model
        torch.cuda.empty_cache()
        df = eval_rows[["path", "battery", "condition", "source_set", "source_id", "category",
                        "split", "variant"]].copy()
        y = df.category.map(C.CAT_TO_IDX).to_numpy()
        df["pred"] = [C.CATEGORIES[i] for i in p16.argmax(1)]
        df["correct"] = (p16.argmax(1) == y).astype(int)
        df["p_true"] = p16[np.arange(len(y)), y]
        for i, c in enumerate(C.CATEGORIES):
            df[f"p_{c}"] = p16[:, i]
        df.to_csv(out, index=False)
        preds[name] = df
        print(f"  {name}: {time.time() - t0:0.0f} s")
    return preds


# =============================================================================
# 10. Equivalence gate
# =============================================================================
COND_ORDER = ["intact", "cutH_intact", "frankenstein", "cutG_intact", "scrambled"]
COND_LABEL = {"intact": "Intact", "cutH_intact": "Cut (H)", "frankenstein": "Frankenstein",
              "cutG_intact": "Cut (grid)", "scrambled": "Scrambled"}
MANIP_CONDS = ["frank_top", "frank_bottom", "scrambled"]
MODEL_LABEL = {"bagnet9": "BagNet-9", "bagnet17": "BagNet-17", "bagnet33": "BagNet-33", "resnet50": "ResNet-50"}
FAMILIES = [("frankenstein", "cutH_intact", "frankenstein"), ("scramble", "cutG_intact", "scrambled"),
            ("cut_cost_H", "intact", "cutH_intact"), ("cut_cost_grid", "intact", "cutG_intact")]


def weak_manipulation_counts(man):
    m = man[man.condition.isin(MANIP_CONDS) & man.battery.isin(BATTERIES)]
    out = {}
    for (b, c, s), g in m.groupby(["battery", "condition", "source_set"]):
        out[f"{b}/{c}/{s}"] = dict(excluded=int((g.manip_strength < CONFIG["MIN_MANIP_STRENGTH"]).sum()),
                                   total=len(g))
    return out


def per_source(df, battery, min_cell=None):
    d = df[df.battery == battery].copy()
    info = man.set_index("path")
    d["manip_strength"] = d.path.map(info["manip_strength"])
    d = d[~(d.condition.isin(MANIP_CONDS) & (d.manip_strength < CONFIG["MIN_MANIP_STRENGTH"]))]
    d["cond"] = d.condition.replace({"frank_top": "frankenstein", "frank_bottom": "frankenstein"})
    if min_cell is not None:
        cells = man[(man.battery == battery) & (man.condition == "cutG_intact")].set_index("source_id")["cell_px"]
        d = d[d.source_id.isin(cells[cells >= min_cell].index)]
    return d.groupby(["source_set", "source_id", "category", "cond"])[["correct", "p_true"]].mean().reset_index()


def balanced_mean(values, cats):
    s = pd.Series(np.asarray(values, float)).groupby(np.asarray(cats)).mean()
    return float(s.mean())


def gate_analysis(preds, min_cell=None, tag="main"):
    rng = np.random.default_rng(C.derive_seed("gate", tag))
    rows, acc_rows = [], []
    floor = CONFIG["FLOOR_MULT"] * C.CHANCE_16
    for name, df in preds.items():
        for battery in BATTERIES:
            ps = per_source(df, battery, min_cell=min_cell)
            if ps.empty:
                continue
            sets = {s: ps[ps.source_set == s] for s in sorted(ps.source_set.unique())}
            if len(sets) > 1:
                sets["pooled"] = ps
            for sset, g in sets.items():
                wide_acc = g.pivot(index="source_id", columns="cond", values="correct")
                wide_p = g.pivot(index="source_id", columns="cond", values="p_true")
                cat_of = g.drop_duplicates("source_id").set_index("source_id")["category"]
                for cond in COND_ORDER:
                    if cond in wide_acc:
                        m, lo, hi = C.bootstrap_mean_ci(wide_acc[cond], CONFIG["N_BOOT"], rng=rng)
                        acc_rows.append(dict(model=name, battery=battery, source_set=sset, condition=cond,
                                             acc=m, lo=lo, hi=hi, n=int(wide_acc[cond].notna().sum())))
                for fam, ctrl, manip in FAMILIES:
                    if ctrl not in wide_acc or manip not in wide_acc:
                        continue
                    both = wide_acc[[ctrl, manip]].dropna().index
                    if len(both) < 5:
                        continue
                    d_acc = (wide_acc.loc[both, ctrl] - wide_acc.loc[both, manip]).to_numpy()
                    d_p = (wide_p.loc[both, ctrl] - wide_p.loc[both, manip]).to_numpy()
                    m, lo, hi = C.bootstrap_mean_ci(d_acc, CONFIG["N_BOOT"], rng=rng)
                    mp, lop, hip = C.bootstrap_mean_ci(d_p, CONFIG["N_BOOT"], rng=rng)
                    cm, clo, chi = C.bootstrap_mean_ci(wide_acc.loc[both, ctrl], CONFIG["N_BOOT"], rng=rng)
                    floor_ok = bool(clo > floor)
                    decision = C.equivalence_decision(lo, hi, m, CONFIG["EQ_BOUND"]) if floor_ok else "floor"
                    rows.append(dict(
                        model=name, battery=battery, source_set=sset, family=fam, n_sources=len(both),
                        ctrl_acc=cm, ctrl_acc_lo=clo, ctrl_acc_hi=chi,
                        drop_acc=m, drop_acc_lo=lo, drop_acc_hi=hi,
                        drop_acc_cat_balanced=balanced_mean(d_acc, cat_of.loc[both]),
                        drop_ptrue=mp, drop_ptrue_lo=lop, drop_ptrue_hi=hip,
                        dz=C.cohens_dz(d_acc), p_signflip=C.sign_flip_pvalue(d_acc, CONFIG["N_BOOT"], rng=rng),
                        drop_sig=bool(lo > 0 or hi < 0), floor_ok=floor_ok, decision=decision))
    return pd.DataFrame(rows), pd.DataFrame(acc_rows)


def family_status(f):
    """f: rows for one model, battery and family, indexed by source_set."""
    key = "pooled" if "pooled" in f.index else f.index[0]
    if not f.loc[key, "floor_ok"]:
        return "INCONCLUSIVE_FLOOR"
    if (f.decision == "non-equivalent").any():
        return "FAIL"
    if f.loc[key, "decision"] == "equivalent":
        return "PASS"
    return "INCONCLUSIVE_WIDE_CI"


def gate_verdict(drops, flags):
    out = {}
    prim = CONFIG["PRIMARY_LOCAL"]
    for battery in BATTERIES:
        b = drops[(drops.model == prim) & (drops.battery == battery)]
        if b.empty:
            continue
        res = {}
        for fam in ["frankenstein", "scramble"]:
            f = b[b.family == fam].set_index("source_set")
            if f.empty:
                continue
            res[fam] = dict(status=family_status(f),
                            by_set={s: dict(decision=r.decision, floor_ok=bool(r.floor_ok),
                                            ctrl_acc=round(float(r.ctrl_acc), 3), drop=round(float(r.drop_acc), 4),
                                            ci=[round(float(r.drop_acc_lo), 4), round(float(r.drop_acc_hi), 4)])
                                    for s, r in f.iterrows()})
        hr = drops[(drops.model == "resnet50") & (drops.battery == battery)]
        hr = hr[hr.source_set == ("pooled" if (hr.source_set == "pooled").any() else hr.source_set.iloc[0])] if not hr.empty else hr
        res["resnet50_headroom"] = {r.family: dict(ctrl_acc=round(float(r.ctrl_acc), 3), drop=round(float(r.drop_acc), 4),
                                                   ci=[round(float(r.drop_acc_lo), 4), round(float(r.drop_acc_hi), 4)],
                                                   drop_sig=bool(r.drop_sig))
                                    for r in hr.itertuples()}
        out[battery] = res
    prim_b = CONFIG["PRIMARY_BATTERY"] if CONFIG["PRIMARY_BATTERY"] in out else "configural"
    behavioural = all(out[prim_b][f]["status"] == "PASS" for f in ["frankenstein", "scramble"] if f in out[prim_b])
    sil_flags = [f for f in flags if f.startswith("configural/")]
    tex_flags = [f for f in flags if f.startswith("configural_tex/")]
    out["primary_battery"] = prim_b
    out["silhouette_local_equivalence"] = "ANALYTIC_PASS" if not sil_flags else "FLAGGED"
    out["overall"] = "PASS" if behavioural and not sil_flags and not tex_flags else "NOT PASSED"
    return out


def latex_table(drops, path):
    d = drops[drops.family.isin(["frankenstein", "scramble"])].copy()
    fam = {"frankenstein": "Frankenstein", "scramble": "Grid scramble"}
    sset = {"geirhos": "Geirhos", "coco": "COCO", "pooled": "Pooled"}
    lines = [r"\begin{tabular}{lllllrl}", r"\hline",
             r"Rendering & Model & Manipulation & Set & $n$ & Drop [95\% CI] & Decision \\", r"\hline"]
    for r in d.sort_values(["battery", "model", "family", "source_set"]).itertuples():
        lines.append(f"{BATTERY_LABEL[r.battery]} & {MODEL_LABEL.get(r.model, r.model)} & {fam[r.family]} & "
                     f"{sset.get(r.source_set, r.source_set)} & {r.n_sources} & "
                     f"{r.drop_acc:0.3f} [{r.drop_acc_lo:0.3f}, {r.drop_acc_hi:0.3f}] & {r.decision} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines))


# =============================================================================
# 11. Figures
# =============================================================================
def montage(rel_paths, n_rows, n_cols, pad=6):
    tile = H
    canvas = np.full((n_rows * tile + (n_rows + 1) * pad, n_cols * tile + (n_cols + 1) * pad, 3), 0.75)
    for k, rel in enumerate(rel_paths):
        if rel is None:
            continue
        i, j = divmod(k, n_cols)
        a = np.asarray(Image.open(C.STIM_LOCAL / rel).convert("RGB"), dtype=np.float64) / 255
        y0, x0 = pad + i * (tile + pad), pad + j * (tile + pad)
        canvas[y0:y0 + tile, x0:x0 + tile] = a
    return canvas, tile, pad


def fig_examples(man):
    C.set_pub_style()
    conds = ["intact", "cutH_intact", "frank_top", "frank_bottom", "cutG_intact", "scrambled"]
    labels = ["Intact", "Cut (H)", "Frank. top", "Frank. bottom", "Cut (grid)", "Scrambled"]
    for battery in BATTERIES:
        cfg = man[man.battery == battery]
        if cfg.empty:
            continue
        pool = cfg[(cfg.source_set != "geirhos") & (cfg.split == "test")]
        pool = pool if not pool.empty else cfg[cfg.split == "test"]
        rng = np.random.default_rng(C.derive_seed("fig_examples"))
        cats = rng.choice(sorted(pool.category.unique()), size=min(4, pool.category.nunique()), replace=False)
        rels = []
        for cat in cats:
            sid = sorted(pool[pool.category == cat].source_id.unique())[0]
            for cnd in conds:
                q = cfg[(cfg.source_id == sid) & (cfg.condition == cnd) & (cfg.variant == 0)]
                rels.append(q.path.iloc[0] if len(q) else None)
        img, tile, pad = montage(rels, len(cats), len(conds))
        fig, ax = plt.subplots(figsize=(2.2 * len(conds), 2.2 * len(cats)))
        ax.imshow(img)
        ax.set_xticks([pad + j * (tile + pad) + tile / 2 for j in range(len(conds))])
        ax.set_xticklabels(labels)
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="left")
        ax.set_yticks([pad + i * (tile + pad) + tile / 2 for i in range(len(cats))])
        ax.set_yticklabels(cats)
        for s in ax.spines.values():
            s.set_linewidth(C.SPINE_LW)
        C.save_figure(fig, FIGS / f"stage1_examples_{battery}")
        plt.close(fig)

    nav = man[man.battery == "navon"]
    ov = man[man.battery == "overlap"]
    pick = lambda df, cond: df[df.condition == cond].sort_values("path").path.iloc[0] if (df.condition == cond).any() else None
    rels = [pick(nav, c) for c in ["navon_congruent", "navon_incongruent", "navon_global_only", "navon_local_only"]]
    pid = ov[ov.condition == "overlap_large"].sort_values("path").source_id.iloc[0] if (ov.condition == "overlap_large").any() else None
    for c in ["outline_single", "overlap_large", "overlap_small", "side_by_side"]:
        if c == "outline_single" or pid is None:
            rels.append(pick(ov, c))
        else:
            q = ov[(ov.condition == c) & (ov.source_id == pid)]
            rels.append(q.path.iloc[0] if len(q) else None)
    img, tile, pad = montage(rels, 2, 4)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.imshow(img)
    ax.set_xticks([pad + j * (tile + pad) + tile / 2 for j in range(4)])
    ax.set_xticklabels(["Congruent / Single", "Incongruent / Overlap L", "Global only / Overlap S",
                        "Local only / Side by side"])
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="left")
    ax.set_yticks([pad + tile / 2, 2 * pad + 1.5 * tile])
    ax.set_yticklabels(["Navon", "Overlapping"])
    for s in ax.spines.values():
        s.set_linewidth(C.SPINE_LW)
    C.save_figure(fig, FIGS / "stage1_navon_overlap_examples")
    plt.close(fig)


def fig_gate(acc):
    """Accuracy per condition, silhouette (a) and texture (b) renderings, pooled sources."""
    C.set_pub_style()
    batteries = [b for b in BATTERIES if (acc.battery == b).any()]
    models = [m for m in CONFIG["MODELS"] if m in acc.model.unique()]
    markers = ["o", "s", "D", "^"]
    colours = ["#9ecae1", "#4292c6", "#08519c", "#d94801"]
    fig, axes = plt.subplots(1, len(batteries), figsize=(9.5 * len(batteries), 5.8), squeeze=False)
    width = 0.16
    x = np.arange(len(COND_ORDER))
    for p, (ax, battery) in enumerate(zip(axes[0], batteries)):
        a = acc[acc.battery == battery]
        sset = "pooled" if (a.source_set == "pooled").any() else a.source_set.iloc[0]
        a = a[a.source_set == sset]
        for k, m in enumerate(models):
            g = a[a.model == m].set_index("condition").reindex(COND_ORDER)
            xs = x + (k - (len(models) - 1) / 2) * width
            ax.errorbar(xs, g.acc, yerr=[g.acc - g.lo, g.hi - g.acc], fmt=markers[k], ms=8, capsize=4,
                        lw=1.5, color=colours[k], label=MODEL_LABEL[m])
        ax.axhline(C.CHANCE_16, ls="--", lw=1.2, color="0.4")
        ax.text(len(COND_ORDER) - 0.55, C.CHANCE_16 + 0.015, "Chance", ha="right", va="bottom",
                fontsize=13, color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER], rotation=20, ha="right")
        ax.set_xlim(-0.5, len(COND_ORDER) - 0.5)
        ax.set_ylabel(f"Accuracy, {BATTERY_LABEL[battery].lower()} (16-way)")
        C.style_axes(ax, ylim=(0, 1), ystep=0.2, decimals=1)
        ax.yaxis.set_label_coords(-0.11, 0.5)
        ax.text(-0.11, 1.04, "ab"[p], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=20, fontweight="bold")
    leg = axes[0, -1].legend(loc="upper right", ncol=2)
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.25)
    C.save_figure(fig, FIGS / "stage1_gate_accuracy")
    plt.close(fig)


# =============================================================================
# 12. Run validation
# =============================================================================
print("\n[3] Figures: examples")
fig_examples(man)

if RUN_MODELS:
    print("\n[4] Model predictions")
    preds = run_predictions(man)

    print("\n[5] Equivalence gate")
    drops, acc = gate_analysis(preds)
    drops_s, _ = gate_analysis(preds, min_cell=CONFIG["MIN_CELL_PX"], tag="sensitivity")
    drops.to_csv(RES / "gate_drops.csv", index=False)
    acc.to_csv(RES / "gate_accuracy.csv", index=False)
    drops_s.to_csv(RES / "gate_drops_sensitivity_cells.csv", index=False)
    latex_table(drops, RES / "gate_table.tex")
    fig_gate(acc)

    sanity = {}
    for name, df in preds.items():
        ref = df[df.condition == "ref_silhouette"].correct.mean()
        ours = df[(df.battery == "configural") & (df.condition == "intact") & (df.source_set == "geirhos")].correct.mean()
        sanity[name] = dict(original_silhouette_acc=round(float(ref), 4), binarised_intact_acc=round(float(ours), 4))

    flags = lowlevel_flags(ll_summary)
    verdict = gate_verdict(drops, flags)
    verdict_s = gate_verdict(drops_s, flags)
    report = dict(version=CONFIG["VERSION"], config_hash=CFG_HASH, eq_bound=CONFIG["EQ_BOUND"],
                  primary_model=CONFIG["PRIMARY_LOCAL"], verdict=verdict,
                  sensitivity_min_cell_px=CONFIG["MIN_CELL_PX"], verdict_sensitivity=verdict_s,
                  lowlevel_flags=flags, binarisation_sanity=sanity,
                  weak_manipulations_excluded=weak_manipulation_counts(man),
                  n_sources={f"{b}/{s}": int(((man.battery == b) & (man.condition == "intact")
                                              & (man.source_set == s)).sum())
                             for b in BATTERIES for s in ["geirhos", "coco"]})
    C.save_json(report, RES / "gate_report.json")

    pd.set_option("display.width", 220)
    cols = ["model", "battery", "source_set", "family", "n_sources", "ctrl_acc", "drop_acc", "drop_acc_lo",
            "drop_acc_hi", "drop_acc_cat_balanced", "decision"]
    print(drops[drops.family.isin(["frankenstein", "scramble"])][cols].round(3).to_string(index=False))
    print("\nCut cost (intact minus cut control):")
    cc = drops[drops.family.str.startswith("cut_cost")]
    cc = cc[(cc.source_set == "pooled") | ((cc.battery == "configural_tex") & (cc.source_set == "coco"))]
    print(cc[["model", "battery", "family", "ctrl_acc", "drop_acc", "drop_acc_lo", "drop_acc_hi"]].round(3).to_string(index=False))
    print("\nBinarisation sanity:", json.dumps(sanity))
    print("Low-level flags:", flags or "none")
    print("\n" + "=" * 78)
    for battery in BATTERIES:
        if battery not in verdict:
            continue
        for fam in ["frankenstein", "scramble"]:
            if fam in verdict[battery]:
                print(f"  {BATTERY_LABEL[battery]:10s} {fam:13s} {verdict[battery][fam]['status']:22s}"
                      f"(cells >= {CONFIG['MIN_CELL_PX']} px: {verdict_s[battery][fam]['status']})")
        print(f"  {'':10s} ResNet-50 headroom: " + ", ".join(
            f"{k} {v['drop']:0.3f} [{v['ci'][0]:0.3f}, {v['ci'][1]:0.3f}]"
            for k, v in verdict[battery]["resnet50_headroom"].items() if not k.startswith("cut_cost")))
    print(f"  Silhouette local equivalence (image statistics): {verdict['silhouette_local_equivalence']}")
    print(f"  Behavioural gate evaluated on: {BATTERY_LABEL[verdict['primary_battery']]}")
    print(f"  STAGE 1 GATE: {verdict['overall']}")
    print("=" * 78)

print("\nDone. Outputs in", RES, "and", FIGS)
