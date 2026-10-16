"""
STAGE 6b: Does zero ablation break the readout? (exploratory, reads Stage 6 outputs only)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (no GPU needed; a few minutes).
No model is run and no image is loaded: everything comes from the decisions and probe predictions
saved by Stage 6 (results/stage6/checkpoints). Exploratory; it carries no confirmatory claim.

Question. In Stage 6, linear probes retrained on the lesioned network's penultimate features still read
category almost as well as before, yet the network's own output fell to the floor under zero ablation,
and its errors moved away from BagNet-33's (H5, wrong direction). Under mean ablation the same lesions
moved errors towards BagNet-33's. The proposed explanation: zeroing channels pushes activity outside the
training distribution, so the fixed final layer collapses onto a few categories (readout failure), while
the information itself survives in the features.

Predictions, written before this script is run
  P1  Under zero ablation the configural lesion concentrates the network's decisions on hold-out grid
      controls: the share of decisions going to the single most frequent category, and 1 minus the
      normalised entropy of the decisions, are higher than intact and higher than under mean ablation.
  P2  Across all lesions (configural, local, 100 damage-matched random; each ablation method), the change
      in Jensen-Shannon distance to BagNet-33's errors rises with that concentration (positive Spearman).
  P3  The readout gap, probe accuracy minus the network's own accuracy on the same grid controls, is
      larger under zero than under mean ablation for the configural lesion.
  Any prediction that fails is reported as failed.

Measures (hold-out grid controls, category-balanced where a mean is taken)
  top share      fraction of decisions on the most frequent predicted category (1/16 = uniform)
  concentration  1 - entropy(decision distribution) / log2(16)
  accuracy       the network's own 16-way accuracy
  probe          accuracy of the grid-control category probe retrained under the same lesion
  readout gap    probe - accuracy
  JS             Jensen-Shannon distance between off-diagonal error distributions and BagNet-33's
Intervals: category-stratified bootstrap over hold-out objects (10,000 resamples) for intact and the
configural and local lesions; random lesions are summarised by mean and 95% range across the 100 sets.

Outputs: results/stage6b/*.csv|json|tex, figures/stage6b/*.pdf|png
"""

import os
import sys
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

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
sfx = "_smoke" if SMOKE else ""
PLAN = C.load_json(C.FROZEN_PLAN)["plan"]
SETS = C.load_json(C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{sfx}.json")
CONFIG = dict(MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]], REFERENCE=PLAN["models"]["local_reference"],
              EXCLUDED=PLAN["data"]["holdout_categories_excluded"], MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
              MIN_OBJECTS_PER_CATEGORY=2 if SMOKE else 20, N_BOOT=300 if SMOKE else 10000)
CK = C.RESULTS_DIR / "stage6" / "checkpoints"
RES = C.RESULTS_DIR / "stage6b"
FIGS = C.FIG_DIR / "stage6b"
for d in [RES, FIGS]:
    d.mkdir(parents=True, exist_ok=True)
print("Stage 6b (exploratory) | reads Stage 6 checkpoints only; no model is run")

# hold-out object order exactly as in Stage 6 (manifest only, no images)
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
d = hm[(hm.battery == "configural_tex") & hm.condition.isin(["cutG_intact", "scrambled"])].copy()
d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_STRENGTH"]))]
d = d[~d.category.isin(CONFIG["EXCLUDED"])]
keep = set(d[d.condition == "cutG_intact"].source_id) & set(d[d.condition == "scrambled"].source_id)
d = d[d.source_id.isin(keep)]
n_per = d.drop_duplicates("source_id").category.value_counts()
d = d[~d.category.isin(n_per[n_per < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
OBJECTS = sorted(d.source_id.unique())
CAT = d.drop_duplicates("source_id").set_index("source_id").loc[OBJECTS, "category"].to_numpy()
Y = np.array([C.CAT_TO_IDX[c] for c in CAT])
if SMOKE:
    z = np.load(CK / f"{CONFIG['REFERENCE']}_intact{sfx}.npz")
    assert len(z["dec_grid"]) == len(OBJECTS), "object order differs from Stage 6"


def load(name, lab):
    f = CK / f"{name}_{lab.replace(' ', '_').replace('(', '').replace(')', '')}{sfx}.npz"
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


REF = load(CONFIG["REFERENCE"], "intact")["dec_grid"]
assert len(REF) == len(OBJECTS), "Stage 6 checkpoints do not match the hold-out object list"

# category-stratified bootstrap weights over objects
rng = np.random.default_rng(C.derive_seed("stage6b_boot"))
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
OFFDIAG = ~np.eye(C.N_CAT, dtype=bool).ravel()


def onehot(dec):
    E = np.zeros((len(dec), C.N_CAT), np.float32)
    E[np.arange(len(dec)), dec] = 1
    return E


def joint(dec):
    E = np.zeros((len(dec), C.N_CAT * C.N_CAT), np.float32)
    E[np.arange(len(dec)), Y * C.N_CAT + dec] = 1
    return E


def js(P, Q):
    P, Q = P[:, OFFDIAG], Q[:, OFFDIAG]
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
    M = 0.5 * (P + Q)
    kl = lambda a, b: np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0).sum(1)
    return np.sqrt(np.clip(0.5 * kl(P, M) + 0.5 * kl(Q, M), 0, None))


REF_J0, REF_JB = (W0 @ joint(REF))[None], W @ joint(REF)


def metrics(o, boot=False):
    """Point values (and bootstrap arrays) of every measure for one condition."""
    w = [W0[None]] + ([W] if boot else [])
    out = []
    for ww in w:
        dist = ww @ onehot(o["dec_grid"])                       # category-balanced decision distribution
        dist = dist / dist.sum(1, keepdims=True)
        top = dist.max(1)
        ent = -(np.where(dist > 0, dist * np.log2(np.maximum(dist, 1e-12)), 0)).sum(1)
        acc = ww @ (o["dec_grid"] == Y).astype(np.float32)
        probe = ww @ (o["catgrid_pred"] == Y).astype(np.float32) if "catgrid_pred" in o else np.full(len(ww), np.nan)
        ref = REF_J0 if ww.shape[0] == 1 else REF_JB
        j = js(ww @ joint(o["dec_grid"]), ref)
        out.append(dict(top_share=top, concentration=1 - ent / np.log2(C.N_CAT), accuracy=acc, probe=probe,
                        readout_gap=probe - acc, js=j))
    return out


ci = lambda b: (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)))
rows, lesion_rows, contrast_rows = [], [], []
for name in CONFIG["MODELS"]:
    e = SETS["models"][name]["sites"][SETS["models"][name]["primary_site"]]
    n_dm = len(e["random_damage_matched"])
    intact = load(name, "intact")
    mi, mib = metrics(intact, boot=True)
    for k in mi:
        rows.append(dict(model=name, ablation="none", condition="intact", measure=k, value=float(mi[k][0]),
                         lo=ci(mib[k])[0], hi=ci(mib[k])[1]))
    for mode, tag in [("zero", ""), ("mean", " (mean)")]:
        conf = load(name, f"configural{tag}")
        mc, mcb = metrics(conf, boot=True)
        loc = load(name, f"local{tag}")
        ml, mlb = metrics(loc, boot=True)
        rnd = [metrics(load(name, f"damage {i + 1}{tag}"))[0] for i in range(n_dm)]
        for lab, m, mb in [("configural", mc, mcb), ("local", ml, mlb)]:
            for k in m:
                rows.append(dict(model=name, ablation=mode, condition=lab, measure=k, value=float(m[k][0]),
                                 lo=ci(mb[k])[0], hi=ci(mb[k])[1]))
        for k in mc:
            v = np.array([r[k][0] for r in rnd])
            rows.append(dict(model=name, ablation=mode, condition="random (damage-matched)", measure=k,
                             value=float(np.nanmean(v)), lo=float(np.nanquantile(v, 0.025)),
                             hi=float(np.nanquantile(v, 0.975))))
        for lab, m in [("configural", mc), ("local", ml)] + [(f"damage {i + 1}", r) for i, r in enumerate(rnd)]:
            lesion_rows.append(dict(model=name, ablation=mode, lesion=lab,
                                    **{k: float(v[0]) for k, v in m.items()},
                                    js_change=float(m["js"][0] - mi["js"][0])))
        # P1 and P3 contrasts for the configural lesion (paired bootstrap)
        for k, lab in [("top_share", "P1 top share, configural minus intact"),
                       ("concentration", "P1 concentration, configural minus intact"),
                       ("readout_gap", "P3 readout gap, configural minus intact")]:
            dlt = mcb[k] - mib[k]
            contrast_rows.append(dict(model=name, ablation=mode, contrast=lab, estimate=float(mc[k][0] - mi[k][0]),
                                      lo=ci(dlt)[0], hi=ci(dlt)[1]))
        if mode == "mean":
            zc, zcb = metrics(load(name, "configural"), boot=True)
            for k, lab in [("top_share", "P1 top share, zero minus mean ablation"),
                           ("concentration", "P1 concentration, zero minus mean ablation"),
                           ("readout_gap", "P3 readout gap, zero minus mean ablation")]:
                dlt = zcb[k] - mcb[k]
                contrast_rows.append(dict(model=name, ablation="zero vs mean", contrast=lab,
                                          estimate=float(zc[k][0] - mc[k][0]), lo=ci(dlt)[0], hi=ci(dlt)[1]))

res, les, con = pd.DataFrame(rows), pd.DataFrame(lesion_rows), pd.DataFrame(contrast_rows)
# P2: across lesions, JS change vs concentration (Spearman, bootstrap over lesions)
p2 = []
for name in CONFIG["MODELS"]:
    for mode in ["zero", "mean", "both"]:
        g = les[(les.model == name) & ((les.ablation == mode) if mode != "both" else True)]
        r = stats.spearmanr(g.concentration, g.js_change).correlation
        bs = []
        rr = np.random.default_rng(C.derive_seed("stage6b_p2", name, mode))
        for _ in range(2000 if not SMOKE else 200):
            idx = rr.integers(0, len(g), len(g))
            bs.append(stats.spearmanr(g.concentration.to_numpy()[idx], g.js_change.to_numpy()[idx]).correlation)
        p2.append(dict(model=name, ablation=mode, n_lesions=len(g), spearman=float(r),
                       lo=float(np.nanquantile(bs, 0.025)), hi=float(np.nanquantile(bs, 0.975))))
p2 = pd.DataFrame(p2)

# verdicts
verd = []
for name in CONFIG["MODELS"]:
    c = con[con.model == name]
    p1 = bool(c[(c.ablation == "zero") & (c.contrast == "P1 top share, configural minus intact")].lo.iloc[0] > 0
              and c[(c.contrast == "P1 top share, zero minus mean ablation")].lo.iloc[0] > 0)
    p2v = bool(p2[(p2.model == name) & (p2.ablation == "both")].lo.iloc[0] > 0)
    p3 = bool(c[c.contrast == "P3 readout gap, zero minus mean ablation"].lo.iloc[0] > 0)
    verd.append(dict(model=name, P1=p1, P2=p2v, P3=p3))
verd = pd.DataFrame(verd)

res.to_csv(RES / f"readout_measures{sfx}.csv", index=False)
les.to_csv(RES / f"per_lesion{sfx}.csv", index=False)
con.to_csv(RES / f"contrasts{sfx}.csv", index=False)
p2.to_csv(RES / f"js_vs_concentration{sfx}.csv", index=False)
C.save_json(dict(verdicts=verd.to_dict(orient="records"), run_at=time.strftime("%Y-%m-%d %H:%M:%S")),
            RES / f"stage6b_summary{sfx}.json")

lines = [r"\begin{tabular}{llrrrr}", r"\hline",
         r"Model & Condition & Network accuracy & Probe accuracy & Readout gap & Top-category share \\", r"\hline"]
for name in CONFIG["MODELS"]:
    for abl, cond in [("none", "intact"), ("zero", "configural"), ("mean", "configural"), ("zero", "random (damage-matched)"),
                      ("mean", "random (damage-matched)")]:
        g = res[(res.model == name) & (res.ablation == abl) & (res.condition == cond)].set_index("measure")
        lab = cond if abl == "none" else f"{cond}, {abl} ablation"
        lines.append(f"{C.MODEL_INFO[name][0]} & {lab} & {g.loc['accuracy', 'value']:0.3f} & {g.loc['probe', 'value']:0.3f} & "
                     f"{g.loc['readout_gap', 'value']:0.3f} & {g.loc['top_share', 'value']:0.3f} \\\\")
lines += [r"\hline", r"\end{tabular}"]
(RES / "stage6b_table.tex").write_text("\n".join(lines))


def tidy(ax, axis, dec):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{dec}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def figure():
    """(a, b) Concentration vs change in JS distance to BagNet-33 errors across all lesions, per model."""
    C.set_pub_style()
    fig, axes = plt.subplots(1, len(CONFIG["MODELS"]), figsize=(8.5 * len(CONFIG["MODELS"]), 7))
    def bounds(v):
        lo, hi = min(0.0, float(np.nanmin(v))), max(0.0, float(np.nanmax(v)))
        pad = 0.04 * max(hi - lo, 1e-6)
        lo, hi = lo - (pad if lo < 0 else 0), hi + pad
        step = next(st for st in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if (hi - lo) / st <= 6)
        return np.floor(lo / step) * step, np.ceil(hi / step) * step, step

    xlo, xhi, xs = 0.0, 1.0, 0.2                    # concentration is bounded by 0 and 1
    ylo, yhi, ys = bounds(les.js_change.to_numpy())
    for p, (ax, name) in enumerate(zip(axes, CONFIG["MODELS"])):
        for mode, mk, mfc in [("zero", "o", None), ("mean", "o", "white")]:
            g = les[(les.model == name) & (les.ablation == mode)]
            r = g[g.lesion.str.startswith("damage")]
            ax.scatter(r.concentration, r.js_change, s=40, marker=mk, facecolors=mfc or "#8a8a8a", edgecolors="#8a8a8a",
                       lw=1.2, label=f"Random lesions, {mode} ablation" if p == 0 else None, zorder=2)
            cf = g[g.lesion == "configural"].iloc[0]
            ax.scatter(cf.concentration, cf.js_change, s=180, marker="*", facecolors=mfc or "#2a78d6",
                       edgecolors="#2a78d6", lw=2, label=f"Configural lesion, {mode} ablation" if p == 0 else None, zorder=3)
        ax.axhline(0, color="#000000", lw=0.8)
        C.style_axes(ax, xlim=(xlo, xhi), xstep=xs, ylim=(ylo, yhi), ystep=ys, decimals=2 if min(xs, ys) < 0.1 else 1)
        tidy(ax, "x", 2 if xs < 0.1 else 1)
        tidy(ax, "y", 2 if ys < 0.1 else 1)
        ax.set_xlabel("Concentration of decisions\n(1 - normalised entropy)")
        ax.set_ylabel(f"Change in JS distance to BagNet-33 errors\n{C.MODEL_INFO[name][0]}")
        ax.yaxis.set_label_coords(-0.14, 0.5)
        ax.text(-0.14, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.35, bottom=0.27)
    C.save_figure(fig, FIGS / f"stage6b_readout{sfx}")
    plt.close(fig)


figure()

pd.set_option("display.width", 240)
show = lambda df: df.round(4).astype(object).where(df.notna(), "").to_string(index=False)
print("\nReadout measures on hold-out grid controls (intact, configural, local; random = mean and 95% range of sets)")
key = res[res.measure.isin(["accuracy", "probe", "readout_gap", "top_share", "concentration", "js"])]
print(show(key.pivot_table(index=["model", "ablation", "condition"], columns="measure", values="value", sort=False).reset_index()))
print("\nContrasts (P1, P3; paired object bootstrap)")
print(show(con))
print("\nP2: Spearman correlation across lesions, concentration vs change in JS distance")
print(show(p2))
print("\nVerdicts (exploratory)")
print(show(verd))
print("\nDone. Outputs in", RES, "and", FIGS)
