"""
STAGE 4a: Piece interactions (exploratory, calibration test half)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 20 to 30 minutes on an A100).

Question: do the configural channels combine object parts non-additively, and does the configural
lesion make the network's category evidence more additive (parts summed, wholes ignored)?

Method (exact, no sampling)
  Each grid-control image of the calibration TEST half is cut into its non-empty 3 x 3 grid pieces
  (n = 3 to 9). The image is shown with every subset S of the pieces present (2^n images; absent
  pieces are replaced by the white background the stimuli are drawn on). For a response v(S), the
  Walsh (Boolean Fourier) decomposition over the uniform distribution on subsets splits the variance
  of v into orders: order 1 = additive piece effects, order k >= 2 = interactions among k pieces
  (functional ANOVA; Hooker, 2007; O'Donnell, 2014). This is the variance-based counterpart of the
  Shapley and Harsanyi interaction indices (Grabisch & Roubens, 1999; Lundberg et al., 2020;
  Ren et al., 2023). With every subset evaluated, the decomposition is exact.

  Non-additivity  NA = 1 - (order-1 variance / total variance), from 0 (purely additive) to 1.

  A. Unit level. v(S) = spatial mean of a channel's activation at the primary site. Variance at each
     order is summed over objects before taking ratios (variance-weighted pooling). Groups as in
     Stage 4: configural, local, other (same 200-channel sample). NA is also correlated with the
     channel's A1 score (I_CR, localizer half) across all channels at the site.
  B. Network level. v(S) = category evidence: the mean logit of the object's true category minus the
     mean over the 16 categories (invariant to shifts common to all logits). NA per object, for the
     intact network, the configural lesion, the local lesion and the first 20 damage-matched random
     lesions (zero ablation, sets from lesion_sets_A1.json).
  Reference: BagNet-33 (layer4 channels; class evidence), which by construction sums local evidence.

Nothing touches the hold-out set. Everything here is exploratory.
Outputs: results/stage4a/*.csv|json|tex, figures/stage4a/*.pdf|png
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
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = False      # full fp32: interaction terms are small differences
torch.backends.cudnn.allow_tf32 = False

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
CONFIG = dict(
    MIN_PIECES=3,
    MAX_PIECES=9,
    MIN_PIECE_PX=40,
    N_OTHER=200,
    N_RANDOM=20,
    N_BOOT=10000,
    BATCH=128,
    GRID=3,
    MIN_STRENGTH=0.10,
    REFERENCE="bagnet33",
)
if SMOKE:
    CONFIG.update(MAX_PIECES=5, N_OTHER=20, N_RANDOM=3, N_BOOT=300, BATCH=32)

C.ensure_dirs()
C.set_seed()
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage4a"
FIGS = C.FIG_DIR / "stage4a"
for d in [RES, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
sfx = "_smoke" if SMOKE else ""
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{sfx}.json"
assert SETS_PATH.exists(), "lesion_sets_A1.json not found; run Stage 3b first."
SETS = C.load_json(SETS_PATH)
SCORES = pd.read_csv(C.RESULTS_DIR / "stage3" / f"channel_scores_A1{sfx}.csv")
H = W = C.IMG_SIZE
print(f"Stage 4a (exploratory) | sets {SETS_PATH.name} | amendment {SETS.get('amendment')} | device {DEVICE}")

# =============================================================================
# 1. Calibration test half: grid-control images and their pieces
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("frozen_calib")
    C.unpack_stimuli("stage1")
fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
grid = fc[(fc.battery == "configural_tex") & (fc.split == "test") & (fc.condition == "cutG_intact")]
scr = fc[(fc.battery == "configural_tex") & (fc.split == "test") & (fc.condition == "scrambled")
         & (fc.manip_strength >= CONFIG["MIN_STRENGTH"])]
grid = grid[grid.source_id.isin(set(scr.source_id))]          # same objects as Stages 3b and 4
assert not grid.path.map(C.is_holdout_path).any()
grid = grid.drop_duplicates("source_id").sort_values("source_id").reset_index(drop=True)

masks = {}
for f in sorted((C.STIM_LOCAL / "stage1").glob("masks_*.npz")):
    z = np.load(f)
    masks.update(dict(zip(z["source_ids"].tolist(), z["masks"])))


def grid_cells(mask):
    """Non-empty grid cells, same geometry as the stimulus generator (Stage 1c grid_geometry)."""
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
            y0, x0 = R0 + i * ch, C0 + j * cw
            if mask[y0:y0 + ch, x0:x0 + cw].sum() >= CONFIG["MIN_PIECE_PX"]:
                cells.append((y0, y0 + ch, x0, x0 + cw))
    return cells


size = 64 if SMOKE else H
OBJ = []
for r in grid.itertuples():
    if r.source_id not in masks:
        continue
    cells = grid_cells(masks[r.source_id])
    if len(cells) < CONFIG["MIN_PIECES"]:
        continue
    img = np.array(Image.open(C.STIM_LOCAL / r.path).convert("RGB").resize((size, size)), dtype=np.uint8)
    s = size / H
    cells = [(int(round(a * s)), int(round(b * s)), int(round(c * s)), int(round(d * s))) for a, b, c, d in cells]
    OBJ.append(dict(source_id=r.source_id, category=r.category, img=img, cells=cells[:CONFIG["MAX_PIECES"]]))
if SMOKE:
    OBJ = OBJ[:24]
n_pieces = np.array([len(o["cells"]) for o in OBJ])
obj_cat = np.array([o["category"] for o in OBJ])
print(f"  test half: {len(OBJ)} objects with >= {CONFIG['MIN_PIECES']} pieces | pieces per object "
      f"{n_pieces.min()} to {n_pieces.max()} (median {int(np.median(n_pieces))}) | "
      f"{int((2 ** n_pieces).sum())} subset images per model")

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def subset_images(o):
    """All 2^n images of object o: bit i of the subset index = piece i present. Returns normalised floats."""
    n = len(o["cells"])
    S = torch.arange(2 ** n, device=DEVICE)
    present = ((S[:, None] >> torch.arange(n, device=DEVICE)) & 1).bool()          # (2^n, n)
    absent = torch.zeros(2 ** n, size, size, dtype=torch.bool, device=DEVICE)
    for i, (y0, y1, x0, x1) in enumerate(o["cells"]):
        absent[~present[:, i], y0:y1, x0:x1] = True
    x = torch.from_numpy(o["img"]).to(DEVICE).permute(2, 0, 1)[None].float().expand(2 ** n, -1, -1, -1).clone()
    x[absent[:, None].expand(-1, 3, -1, -1)] = 255.0
    return (x / 255.0 - MEAN) / STD


def order_variance(v, n):
    """Walsh-Hadamard transform over the subset axis. v: (2^n, K). Returns (n + 1, K): variance at each
    interaction order (index 0 = squared mean, excluded from all variance ratios)."""
    x = v.double()
    for i in range(n):
        x = x.view(2 ** (n - i - 1), 2, 2 ** i, -1)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).view(2 ** n, -1)
    coef2 = (x / 2 ** n) ** 2
    order = torch.tensor([bin(t).count("1") for t in range(2 ** n)], device=v.device)
    out = torch.zeros(n + 1, v.shape[1], dtype=torch.float64, device=v.device)
    out.index_add_(0, order, coef2)
    return out


def na_from(ov):
    """Non-additivity from order variances (orders 1 and up)."""
    tot = ov[1:].sum(0)
    return np.where(tot > 0, 1 - ov[1] / np.maximum(tot, 1e-300), np.nan)


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

    def prefix(self, x, site):
        for i in range(self.index[site] + 1):
            x = self.stages[i][1](x)
        return x

    def suffix(self, a, site):
        for i in range(self.index[site] + 1, len(self.stages)):
            a = self.stages[i][1](a)
        return self.head(a)


def evidence(logits, cat):
    """Category evidence: mean logit over the category's ImageNet classes minus the mean over categories."""
    e = logits.float() @ M16
    return e[:, C.CAT_TO_IDX[cat]] - e.mean(1)


# =============================================================================
# 3. Run
# =============================================================================
@torch.no_grad()
def run_model(name, site, conditions):
    """conditions: dict label -> channel index array (zero-ablated at site). Returns per-channel order
    variances (summed over objects) and per-object network-level order variances per condition."""
    net = Split(name)
    unit_ov, net_ov = None, {k: [] for k in conditions}
    t0, done, total = time.time(), 0, int((2 ** n_pieces).sum())
    for oi, o in enumerate(OBJ):
        n = len(o["cells"])
        x = subset_images(o)
        acts, ev = [], {k: [] for k in conditions}
        for b0 in range(0, len(x), CONFIG["BATCH"]):
            a = net.prefix(x[b0:b0 + CONFIG["BATCH"]], site)
            acts.append(a.mean(dim=(2, 3)))
            for k, ch in conditions.items():
                aa = a
                if len(ch):
                    aa = a.clone()
                    aa[:, torch.as_tensor(ch, device=DEVICE)] = 0
                ev[k].append(evidence(net.suffix(aa, site), o["category"]))
        ov = order_variance(torch.cat(acts), n)
        ov = torch.cat([ov, torch.zeros(CONFIG["MAX_PIECES"] - n, ov.shape[1], dtype=ov.dtype, device=ov.device)])
        unit_ov = ov if unit_ov is None else unit_ov + ov
        for k in conditions:
            net_ov[k].append(order_variance(torch.cat(ev[k])[:, None], n)[:, 0].cpu().numpy())
        done += 2 ** n
        if (oi + 1) % 100 == 0 or oi == len(OBJ) - 1:
            el = time.time() - t0
            print(f"    {name} {site}: {oi + 1}/{len(OBJ)} objects | {el / 60:0.1f} min | "
                  f"ETA {el / done * (total - done) / 60:0.1f} min")
    del net
    torch.cuda.empty_cache()
    return unit_ov.cpu().numpy(), net_ov


def boot_mean(x, rng, strat=None):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    if strat is None:
        b = rng.choice(x[ok], size=(CONFIG["N_BOOT"], ok.sum())).mean(1)
    else:
        groups = [np.where(ok & (strat == s))[0] for s in np.unique(strat[ok])]
        b = np.array([np.concatenate([x[rng.choice(g, len(g))] for g in groups]).mean()
                      for _ in range(CONFIG["N_BOOT"])])
    return float(np.nanmean(x)), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    d = rng.choice(a, size=(CONFIG["N_BOOT"], len(a))).mean(1) - rng.choice(b, size=(CONFIG["N_BOOT"], len(b))).mean(1)
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


unit_rows, group_rows, contrast_rows, corr_rows, obj_rows, net_rows, spectra = [], [], [], [], [], [], {}
models = [(m, SETS["models"][m]["primary_site"]) for m in SETS["models"] if SETS["models"][m]["primary_site"]]
models.append((CONFIG["REFERENCE"], "layer4"))
for name, site in models:
    label = C.MODEL_INFO[name][0]
    print(f"\n[{label}] site {site}")
    rng = np.random.default_rng(C.derive_seed("stage4a", name))
    if name == CONFIG["REFERENCE"]:
        conditions = {"intact": np.array([], int)}
        groups = None
    else:
        e = SETS["models"][name]["sites"][site]
        conf, local = np.array(e["configural"], int), np.array(e.get("local", []), int)
        rest = np.setdiff1d(np.arange(e["n_channels"]), np.union1d(conf, local))
        other = np.sort(np.random.default_rng(C.derive_seed("stage4_other", name)).choice(
            rest, size=min(CONFIG["N_OTHER"], len(rest)), replace=False))       # same sample as Stage 4
        groups = {"configural": conf, "local": local, "other": other}
        conditions = {"intact": np.array([], int), "configural lesion": conf}
        if len(local):
            conditions["local lesion"] = local
        for r, s in enumerate(e["random_damage_matched"][:CONFIG["N_RANDOM"]]):
            conditions[f"random {r + 1}"] = np.array(s, int)
    uov, nov = run_model(name, site, conditions)
    na_c = na_from(uov)
    tot = uov[1:].sum(0)
    mean_order = np.where(tot > 0, (np.arange(1, uov.shape[0])[:, None] * uov[1:]).sum(0) / np.maximum(tot, 1e-300), np.nan)

    # A. unit level
    if groups is None:
        groups_eff = {"all": np.arange(uov.shape[1])}
    else:
        groups_eff = groups
        sc = SCORES[(SCORES.model == name) & (SCORES.site == site)].set_index("channel")
        ok = np.isfinite(na_c)
        icr = sc.loc[np.arange(uov.shape[1]), "I_CR_localizer"].to_numpy()
        rho = stats.spearmanr(na_c[ok], icr[ok]).correlation
        bs = []
        idx_ok = np.where(ok)[0]
        for _ in range(CONFIG["N_BOOT"] // 10):
            ii = rng.choice(idx_ok, len(idx_ok))
            bs.append(stats.spearmanr(na_c[ii], icr[ii]).correlation)
        corr_rows.append(dict(model=name, site=site, n_channels=int(ok.sum()), spearman_NA_ICR=float(rho),
                              lo=float(np.nanquantile(bs, 0.025)), hi=float(np.nanquantile(bs, 0.975))))
    spectra[name] = {}
    for g, ch in groups_eff.items():
        if not len(ch):
            continue
        for c in ch:
            unit_rows.append(dict(model=name, site=site, group=g, channel=int(c), NA=float(na_c[c]),
                                  mean_order=float(mean_order[c]), total_variance=float(tot[c])))
        for metric, vals in [("NA", na_c[ch]), ("mean_order", mean_order[ch])]:
            m, lo, hi = boot_mean(vals, rng)
            group_rows.append(dict(model=name, site=site, group=g, metric=metric, mean=m, lo=lo, hi=hi,
                                   n=int(np.isfinite(vals).sum())))
        frac = uov[1:, ch] / np.maximum(uov[1:, ch].sum(0, keepdims=True), 1e-300)
        spectra[name][g] = np.nanmean(np.where(tot[ch] > 0, frac, np.nan), 1)
    if groups is not None:
        for g in ["local", "other"]:
            if not len(groups[g]):
                continue
            for metric, arr in [("NA", na_c), ("mean_order", mean_order)]:
                a, b = arr[groups["configural"]], arr[groups[g]]
                d, lo, hi = boot_diff(a, b, rng)
                p = stats.mannwhitneyu(a[np.isfinite(a)], b[np.isfinite(b)]).pvalue if np.isfinite(b).sum() > 1 else np.nan
                contrast_rows.append(dict(model=name, metric=metric, contrast=f"configural - {g}", diff=d, lo=lo, hi=hi,
                                          mannwhitney_p=float(p)))

    # B. network level
    per = {}
    for k, lst in nov.items():
        ov = np.stack([np.pad(v, (0, CONFIG["MAX_PIECES"] + 1 - len(v))) for v in lst], 1)   # (orders, objects)
        per[k] = dict(NA=na_from(ov), var=ov[1:].sum(0))
        for oi in range(len(OBJ)):
            obj_rows.append(dict(model=name, condition=k, source_id=OBJ[oi]["source_id"], category=obj_cat[oi],
                                 n_pieces=int(n_pieces[oi]), NA=float(per[k]["NA"][oi]), variance=float(per[k]["var"][oi])))
    base_var = per["intact"]["var"]
    for k in [k for k in per if not k.startswith("random")]:
        m, lo, hi = boot_mean(per[k]["NA"], rng, strat=obj_cat)
        net_rows.append(dict(model=name, condition=k, NA=m, lo=lo, hi=hi,
                             median_variance_ratio=float(np.nanmedian(per[k]["var"] / np.maximum(base_var, 1e-12)))))
    rnd = [k for k in per if k.startswith("random")]
    if rnd:
        means = np.array([np.nanmean(per[k]["NA"]) for k in rnd])
        vr = np.array([np.nanmedian(per[k]["var"] / np.maximum(base_var, 1e-12)) for k in rnd])
        net_rows.append(dict(model=name, condition="random (damage-matched)", NA=float(means.mean()),
                             lo=float(np.quantile(means, 0.025)), hi=float(np.quantile(means, 0.975)),
                             median_variance_ratio=float(np.median(vr)), n_sets=len(rnd)))
        conf_na = np.nanmean(per["configural lesion"]["NA"])
        d = per["configural lesion"]["NA"] - per["intact"]["NA"]
        m, lo, hi = boot_mean(d, rng, strat=obj_cat)
        net_rows.append(dict(model=name, condition="configural lesion - intact", NA=m, lo=lo, hi=hi))
        net_rows.append(dict(model=name, condition="configural lesion: fraction of random sets with lower NA",
                             NA=float((means < conf_na).mean()), n_sets=len(rnd)))
    print(f"  unit NA: " + " | ".join(f"{g} {np.nanmean(na_c[ch]):0.3f}" for g, ch in groups_eff.items() if len(ch))
          + f" || network NA intact {np.nanmean(per['intact']['NA']):0.3f}"
          + (f", configural lesion {np.nanmean(per['configural lesion']['NA']):0.3f}" if "configural lesion" in per else ""))

# =============================================================================
# 4. Save, tables, figures
# =============================================================================
unit_df, grp_df, con_df, cor_df, obj_df, net_df = map(pd.DataFrame, [unit_rows, group_rows, contrast_rows,
                                                                       corr_rows, obj_rows, net_rows])
unit_df.to_csv(RES / f"channel_interactions{sfx}.csv", index=False)
grp_df.to_csv(RES / f"group_interactions{sfx}.csv", index=False)
con_df.to_csv(RES / f"group_contrasts{sfx}.csv", index=False)
cor_df.to_csv(RES / f"na_vs_icr{sfx}.csv", index=False)
obj_df.to_csv(RES / f"object_interactions{sfx}.csv", index=False)
net_df.to_csv(RES / f"network_interactions{sfx}.csv", index=False)
C.save_json(dict(sets=SETS_PATH.name, config=CONFIG, n_objects=len(OBJ),
                 pieces_per_object=dict(pd.Series(n_pieces).value_counts().sort_index().astype(int)),
                 spectra={m: {g: v.tolist() for g, v in s.items()} for m, s in spectra.items()}),
            RES / f"stage4a_summary{sfx}.json")

# LaTeX: unit-level non-additivity by group, and its correlation with I_CR
fmt = lambda m, lo, hi: f"{m:0.3f} [{lo:0.3f}, {hi:0.3f}]" if np.isfinite(m) else "n/a"
lines = [r"\begin{tabular}{lllrl}", r"\hline",
         r"Model & Site & Channels & $n$ & Non-additivity [95\% CI] \\", r"\hline"]
for r in grp_df[grp_df.metric == "NA"].itertuples():
    lines.append(f"{C.MODEL_INFO[r.model][0]} & {r.site} & {r.group} & {r.n} & {fmt(r.mean, r.lo, r.hi)} \\\\")
lines += [r"\hline"]
for r in cor_df.itertuples():
    lines.append(f"{C.MODEL_INFO[r.model][0]} & {r.site} & \\multicolumn{{2}}{{l}}{{Spearman with $I_{{CR}}$}} & "
                 f"{fmt(r.spearman_NA_ICR, r.lo, r.hi)} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage4a_table.tex").write_text("\n".join(lines))

COL = {"configural": "#2a78d6", "local": "#eb6834", "other": "#8a8a8a", "all": "#000000"}
MARK = {"configural": "o", "local": "s", "other": "D"}


def fig_interactions():
    C.set_pub_style()
    study = [m for m, _ in models if m != CONFIG["REFERENCE"]]
    ref = CONFIG["REFERENCE"]
    fig, axes = plt.subplots(1, len(study) + 1, figsize=(8.2 * (len(study) + 1), 6.8))
    K = CONFIG["MAX_PIECES"]
    ks = np.arange(1, K + 1)
    allv = np.concatenate([v[:K] for s_ in spectra.values() for v in s_.values()])
    lo_dec = int(np.floor(np.log10(np.nanmin(allv[allv > 0])))) if np.any(allv > 0) else -4
    lo_dec = max(lo_dec, -6)
    for p, m in enumerate(study):
        ax = axes[p]
        show = lambda v: np.where(v[:K] >= 10.0 ** lo_dec, v[:K], np.nan)
        for g, v in spectra[m].items():
            ax.plot(ks, show(v), marker=MARK[g], ms=8, lw=2, color=COL[g], label=g.capitalize())
        ax.plot(ks, show(spectra[ref]["all"]), ls="--", lw=2, color="#000000", label=C.MODEL_INFO[ref][0])
        C.style_axes(ax, xlim=(1, K), xstep=1)
        ax.set_yscale("log")
        ax.set_ylim(10.0 ** lo_dec, 1)
        ax.set_yticks([10.0 ** k for k in range(lo_dec, 1)])
        ax.set_yticklabels([f"$10^{{{k}}}$" if k < 0 else "1" for k in range(lo_dec, 1)])
        ax.minorticks_off()
        ax.set_xlabel("Interaction order (pieces)")
        ax.set_ylabel(f"Share of variance\n{C.MODEL_INFO[m][0]}, {dict(models)[m]}")
    ax = axes[-1]
    conds = [("intact", "Intact", "o", "#000000"), ("configural lesion", "Configural lesion", "o", COL["configural"]),
             ("local lesion", "Local lesion", "s", COL["local"]),
             ("random (damage-matched)", "Random (damage-matched)", "D", COL["other"])]
    xt, xl, hi_all = [], [], []
    for i, (m, _) in enumerate(models):
        d = net_df[net_df.model == m]
        for j, (k, lab, mk, col) in enumerate(conds):
            r = d[d.condition == k]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i * 5 + j
            ax.errorbar(x, r.NA, yerr=[[r.NA - r.lo], [r.hi - r.NA]], fmt=mk, ms=10, capsize=5, lw=1.8, color=col,
                        mfc=col if k != "intact" else "white", label=lab if i == 0 else None)
            hi_all.append(r.hi)
        xt.append(i * 5 + 1.5 if m != CONFIG["REFERENCE"] else i * 5)
        xl.append(C.MODEL_INFO[m][0])
    top = min(1.0, np.ceil(max(hi_all) / 0.1) * 0.1 + 0.1) if hi_all else 1.0
    C.style_axes(ax, ylim=(0, top), ystep=0.1 if top <= 0.6 else 0.2, decimals=1)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_xlim(-1, len(models) * 5 - 3)
    ax.set_ylabel("Non-additivity of category evidence")
    for p, ax in enumerate(axes):
        ax.yaxis.set_label_coords(-0.13, 0.5)
        ax.text(-0.13, 1.03, "abcd"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[-1].get_legend_handles_labels()
    leg = fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.35, bottom=0.25)
    C.save_figure(fig, FIGS / f"stage4a_interactions{sfx}")
    plt.close(fig)


fig_interactions()

pd.set_option("display.width", 220)
print("\nUnit level: non-additivity and mean interaction order (mean [95% CI])")
print(grp_df.round(3).to_string(index=False))
print("\nConfigural minus other groups")
print(con_df.round(4).to_string(index=False))
print("\nSpearman correlation of channel non-additivity with I_CR (localizer half)")
print(cor_df.round(3).to_string(index=False))
print("\nNetwork level: non-additivity of category evidence (objects; random = mean and 95% range over sets)")
print(net_df.round(3).to_string(index=False))
print("\nDone. Outputs in", RES, "and", FIGS)
