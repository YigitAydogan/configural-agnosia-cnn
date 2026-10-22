"""
STAGE 12: A second ResNet-50 weight set, supervised ImageNet training (exploratory)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU). Single-channel scores are saved per site and
calibration half, lesion sets as soon as they are chosen, hold-out decisions per block; rerunning picks up
where it stopped. At the end the results folder is zipped to /content for download.

Why. Every result so far comes from one pretrained weight set per architecture. A pre-submission review
asked whether the effects depend on that particular weight set. ResNet-50 trained with standard
supervised ImageNet training (torchvision weights) qualified in the original model selection, ranked second
behind the SWSL weights, and shares the architecture of the primary model. It is therefore the closest
available test of whether the result generalises to another weight set. It is not a second random seed of
the SWSL network: training data and objective also differ.

Everything here is exploratory and carries no confirmatory claim. The procedure repeats the planned unit
localisation (Amendment A1) and the H2 and H3 lesion tests exactly, for this network:
  Calibration (no hold-out image is used until the sets are written):
    single-channel zero ablation at layer1 to layer4, scores I_CR on both calibration halves; site reliable
    if the Spearman-Brown corrected split-half correlation is at least 0.7; configural set = channels with
    I_CR reliably above 0 (one-sided category-stratified bootstrap, Benjamini-Hochberg q = 0.05), bounded to
    1 to 40% of the site; primary site = reliable site with the largest summed I_CR; local set, 100
    size-matched and 100 damage-matched random sets by the planned rules.
    Because the planned set of the SWSL network cost too much recognition to test, the matched-cost set of
    the SWSL follow-up is also built: the top k channels of the primary site with damage closest to the
    CORnet-S configural set's, with its own local set and random sets.
    All sets are written to results/stage12/sets_resnet50.json and its SHA-256 printed before any hold-out
    image is used.
  Hold-out (the same 1421 objects), zero ablation first, mean ablation as robustness:
    configural reliance with the floor rule, random sets at floor left out and counted, category-stratified
    bootstrap over objects (10,000 resamples), and the floor-free absolute scramble loss.

Predictions, written before the first run (for the planned set and for the matched-cost set separately)
  T1  dCR(configural) minus mean dCR(random) has upper bound < 0 for size- and damage-matched sets, and
      dCR(configural) lies below the 5th percentile of the damage-matched sets.
  T2  dCR(configural) minus dCR(local) has upper bound < 0.
  A set whose configural lesion drives grid-control accuracy below the floor is reported as not estimable,
  as in the planned analysis.

Outputs: results/stage12/*.csv|json, and /content/stage12_results.zip for download.
"""

# =============================================================================
# 0. Setup and checks
# =============================================================================
import os
import sys
import json
import time
import shutil
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
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
sfx = "_smoke" if SMOKE else ""

PLAN_SHA_REGISTERED = "dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c"
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
    assert sets_sha == SETS_SHA_REGISTERED, "lesion_sets_A1.json differs from the published version."
SETS = C.load_json(SETS_PATH)
cs = SETS["models"]["cornet_s"]["primary_site"]
TARGET = float(SETS["models"]["cornet_s"]["sites"][cs]["damage_configural"])

MODEL = "resnet50"
CONFIG = dict(
    MODEL=MODEL,
    SITES=["layer1", "layer2", "layer3", "layer4"],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    FLOOR=2 * C.CHANCE_16,
    Q=0.05,
    MIN_FRAC=0.01,
    MAX_FRAC=0.40,
    REL_THRESHOLD=0.7,
    N_RANDOM_SIZE=100,
    N_DAMAGE=100,
    DAMAGE_POOL=10000,
    DAMAGE_TOL=0.02,
    N_BOOT=10000,
    BATCH=128,
    MAX_SUFFIX_IMAGES=1024,
    MAX_ELEMENTS=4e8,
)
if SMOKE:
    CONFIG.update(N_BOOT=300, N_RANDOM_SIZE=5, N_DAMAGE=5, DAMAGE_POOL=40, REL_THRESHOLD=-1.0, BATCH=32,
                  MIN_OBJECTS_PER_CATEGORY=2, SITES=["layer3", "layer4"])

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True     # same numerical settings as the earlier lesion analyses
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage12"
CKPT = RES / "checkpoints"
for dd in [RES, CKPT]:
    dd.mkdir(parents=True, exist_ok=True)
print(f"Stage 12 (exploratory) | {MODEL} | plan {FROZEN['plan_sha256'][:12]} | target damage {TARGET:0.4f} | device {DEVICE}")

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
class Split:
    def __init__(self, name):
        import torchvision
        m = torchvision.models.resnet50(weights=None).eval().to(DEVICE) if SMOKE else C.load_model(name, DEVICE)
        self.name = name
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.stages = [("stem", stem), ("layer1", m.layer1), ("layer2", m.layer2), ("layer3", m.layer3), ("layer4", m.layer4)]
        self.head = lambda x: m.fc(torch.flatten(m.avgpool(x), 1))
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


@torch.no_grad()
def p16n(logits):
    """16-way probability: the category averages renormalised to sum to 1, so chance is 1/16 as
    Amendment A1 assumes. The argmax (the model's decision) is identical to p16's."""
    p = p16(logits)
    return p / p.sum(dim=1, keepdim=True)


# =============================================================================
# 3. Single-channel ablation scores (checkpointed per model, site and half)
# =============================================================================
def channel_scores(net, site, half):
    """D[c, i]: drop in true-category probability on image i when channel c is zeroed. Each chunk of
    ablated copies is run in the same call as an unablated copy, so numerical differences between
    batch shapes never appear as lesion effects."""
    ck = CKPT / f"{net.name}_{site}_{half.name}_p16norm{'_smoke' if SMOKE else ''}.npz"
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
            out = p16n(net.suffix(aa.flatten(0, 1), site))
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


@torch.no_grad()
def baseline_p(net, half):
    """True-category probability of the unlesioned network for every image (cached)."""
    ck = CKPT / f"{net.name}_baseline_{half.name}_p16norm{'_smoke' if SMOKE else ''}.npy"
    if ck.exists():
        p = np.load(ck)
        if len(p) == len(half.df):
            return p + (0.3 if SMOKE else 0.0)                      # smoke test only: random weights sit at chance
    y = torch.from_numpy(half.y).to(DEVICE)
    out = np.zeros(len(half.df), np.float32)
    B = CONFIG["BATCH"]
    for b0 in range(0, len(half.df), B):
        sl = slice(b0, min(len(half.df), b0 + B))
        x = norm(half.imgs[sl])
        for _, st in net.stages:
            x = st(x)
        out[sl] = p16n(net.head(x)).gather(1, y[sl][:, None])[:, 0].cpu().numpy()
    np.save(ck, out)
    return out + (0.3 if SMOKE else 0.0)


def cr_p(pg, ps):
    c0 = C.CHANCE_16
    pg, ps = np.asarray(pg, np.float64), np.asarray(ps, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1.0 - (ps - c0) / (pg - c0)
    return np.where(pg - c0 > 1e-6, v, np.nan)


def stratified_weights(half, n_boot, seed):
    """Bootstrap weights over objects, resampling within category (rows sum to 1)."""
    n = len(half.objects)
    W = np.zeros((n_boot, n), np.float32)
    rng = np.random.default_rng(seed)
    rows = np.arange(n_boot)
    for c in np.unique(half.obj_cat):
        members = np.where(half.obj_cat == c)[0]
        draws = members[rng.integers(0, len(members), size=(n_boot, len(members)))]
        np.add.at(W, (np.repeat(rows, len(members)), draws.ravel()), 1.0)
    return W / n


def site_scores(net, site):
    res = {}
    for hname, half in HALVES.items():
        D, A, sum_act, n_act = channel_scores(net, site, half)
        P0 = baseline_p(net, half)
        pg, ps = P0[half.grid_img], P0 @ half.scr_avg                   # (n_obj,)
        d_grid, d_scr = object_level(D, half)                           # (C, n_obj)
        Pg, Ps = pg.mean(), ps.mean()
        I_cr = cr_p(Pg, Ps) - cr_p(Pg - d_grid.mean(1), Ps - d_scr.mean(1))
        res[hname] = dict(I=I_cr, I_old=(d_grid - d_scr).mean(1), local=d_scr.mean(1), A=A,
                          sum_act=sum_act, n_act=n_act, pg=pg, ps=ps, d_grid=d_grid, d_scr=d_scr,
                          CR_intact=float(cr_p(Pg, Ps)))
    loc, h = res["localizer"], HALVES["localizer"]
    assert np.isfinite(loc["CR_intact"]), (f"CR_p undefined at {net.name} {site}: mean grid probability "
                                           f"{loc['pg'].mean():0.4f} is not above chance (1/16)")
    # one-sided, category-stratified bootstrap over localizer objects (Amendment A1)
    W = stratified_weights(h, CONFIG["N_BOOT"], C.derive_seed(net.name, site, "A1_bootstrap"))
    count = np.zeros(len(loc["I"]), np.int64)
    for b0 in range(0, CONFIG["N_BOOT"], 1000):
        Wb = W[b0:b0 + 1000].astype(np.float64)
        Pg_b, Ps_b = Wb @ loc["pg"], Wb @ loc["ps"]
        Dg_b, Ds_b = Wb @ loc["d_grid"].T, Wb @ loc["d_scr"].T
        Ib = cr_p(Pg_b, Ps_b)[:, None] - cr_p(Pg_b[:, None] - Dg_b, Ps_b[:, None] - Ds_b)
        count += (~(Ib > 0)).sum(0)                                     # nan counts against
    p = (1 + count) / (1 + CONFIG["N_BOOT"])
    p = np.where(np.isfinite(loc["I"]), p, 1.0)
    # selectivity (unchanged): spatially averaged response, grid control vs mean over scrambles
    act_grid = loc["A"][h.grid_img]
    act_scr = h.scr_avg.T @ loc["A"]
    ts, ps_ = stats.ttest_rel(act_grid, act_scr, axis=0, alternative="greater")
    ps_ = np.nan_to_num(ps_, nan=1.0)
    r, _ = stats.spearmanr(loc["I"], res["test"]["I"], nan_policy="omit")
    sb = 2 * r / (1 + r) if r > -1 else np.nan
    total = res["localizer"]["n_act"] + res["test"]["n_act"]
    mean_act = (res["localizer"]["sum_act"] + res["test"]["sum_act"]) / total
    return dict(I_loc=loc["I"], I_test=res["test"]["I"], I_old=loc["I_old"], local=loc["local"],
                p=p, q=bh(p), t_sel=ts, q_sel=bh(ps_), spearman=float(r), sb=float(sb), mean_act=mean_act,
                n_channels=len(loc["I"]), CR_p_intact=loc["CR_intact"])


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
    ranked_I = np.nan_to_num(sc["I_loc"], nan=-np.inf)
    order = np.argsort(-ranked_I)
    sig = np.where(sc["q"] < CONFIG["Q"])[0]
    k = len(sig)
    if k < lo:
        chosen, rule = order[:lo], f"fewer than 1% significant ({k}); top 1% taken"
    elif k > hi:
        chosen, rule = order[:hi], f"more than 40% significant ({k}); top 40% taken"
    else:
        chosen, rule = sig[np.argsort(-ranked_I[sig])], f"{k} FDR-significant channels"
    return np.sort(chosen), rule, k


def random_sets(n_channels, k, n_sets, *key):
    rng = np.random.default_rng(C.derive_seed(*key))
    return [np.sort(rng.choice(n_channels, size=k, replace=False)) for _ in range(n_sets)]


def local_set(net, site, sc, target_damage):
    elig = np.where(sc["I_loc"] <= np.nanmedian(sc["I_loc"]))[0]
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
# 7. Unit localisation and lesion sets (calibration only)
# =============================================================================
SETS_OUT = RES / f"sets_resnet50{sfx}.json"
net = None
if SETS_OUT.exists():
    OUT = C.load_json(SETS_OUT)
    print(f"\n[2] lesion sets loaded ({SETS_OUT.name})")
else:
    print(f"\n[2] {MODEL}: single-channel scores")
    net = Split(MODEL)
    site_info, rows = {}, []
    for site in CONFIG["SITES"]:
        sc = site_scores(net, site)
        conf, rule, n_sig = configural_set(sc)
        reliable = bool(sc["sb"] >= CONFIG["REL_THRESHOLD"])
        site_info[site] = dict(sc=sc, conf=conf, rule=rule, n_sig=n_sig, reliable=reliable,
                               sum_I=float(np.nansum(sc["I_loc"][conf])))
        print(f"  {site}: CR_p intact {sc['CR_p_intact']:0.3f} | Spearman-Brown {sc['sb']:0.3f} "
              f"({'reliable' if reliable else 'NOT reliable'}) | configural: {rule} | summed I_CR {site_info[site]['sum_I']:0.3f}")
        rows.append(pd.DataFrame(dict(site=site, channel=np.arange(sc["n_channels"]), I_CR_localizer=sc["I_loc"],
                                      I_CR_test=sc["I_test"], local_score=sc["local"], p_boot=sc["p"], q=sc["q"],
                                      mean_activation=sc["mean_act"])))
    pd.concat(rows).to_csv(RES / f"channel_scores_resnet50{sfx}.csv", index=False)
    reliable_sites = [s for s in CONFIG["SITES"] if site_info[s]["reliable"]]
    assert reliable_sites, "No reliable site; the analysis stops here (reported as such)."
    primary = max(reliable_sites, key=lambda s: site_info[s]["sum_I"])
    si, sc = site_info[primary], site_info[primary]["sc"]
    n = sc["n_channels"]
    print(f"  primary site: {primary} ({len(si['conf'])} of {n} channels)")
    OUT = dict(model=MODEL, primary_site=primary, target_damage=TARGET,
               sites={s: dict(n_channels=site_info[s]["sc"]["n_channels"], reliable=site_info[s]["reliable"],
                              spearman_brown=site_info[s]["sc"]["sb"], configural_rule=site_info[s]["rule"],
                              n_configural=len(site_info[s]["conf"]), sum_I_CR=site_info[s]["sum_I"])
                      for s in CONFIG["SITES"]},
               mean_activation=sc["mean_act"].tolist(), sets={})

    def build(label, conf, d_conf):
        k = len(conf)
        loc, loc_d = local_set(net, primary, sc, d_conf)
        rs = random_sets(n, k, CONFIG["N_RANDOM_SIZE"], MODEL, primary, label, "size_matched")
        dm, dmd, how, n_tried, _ = damage_matched(net, primary, k, d_conf, n)
        print(f"  {label}: {k} channels, damage {d_conf:0.3f} | local {len(loc)} channels, damage {loc_d:0.3f} | "
              f"damage-matched: {how} ({n_tried} sets tried)")
        return dict(configural=np.sort(conf).tolist(), damage_configural=float(d_conf), local=loc.tolist(),
                    damage_local=float(loc_d), random_size_matched=[s.tolist() for s in rs],
                    damage_random_size_matched=damages(net, primary, rs)[0].tolist(),
                    random_damage_matched=[s.tolist() for s in dm], damage_random_damage_matched=dmd.tolist(),
                    damage_matching=dict(rule=how, sets_tried=n_tried))

    d_planned = float(damages(net, primary, [si["conf"]])[0][0])
    OUT["sets"]["planned"] = build("planned", si["conf"], d_planned)
    order = np.argsort(-np.nan_to_num(sc["I_loc"], nan=-np.inf))
    kmax = max(1, int(round(CONFIG["MAX_FRAC"] * n)))
    dmg_k = damages(net, primary, [order[:k] for k in range(1, kmax + 1)])[0]
    k_star = int(np.argmin(np.abs(dmg_k - TARGET))) + 1
    OUT["matched_cost_scan"] = dmg_k.tolist()
    OUT["sets"]["matched cost"] = build("matched cost", order[:k_star], float(dmg_k[k_star - 1]))
    OUT["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.save_json(OUT, SETS_OUT)
out_sha = hashlib.sha256(SETS_OUT.read_bytes()).hexdigest()
PRIMARY = OUT["primary_site"]
print(f"  sets file SHA-256 {out_sha} | primary site {PRIMARY} | planned {len(OUT['sets']['planned']['configural'])} channels, "
      f"matched cost {len(OUT['sets']['matched cost']['configural'])} channels")

# =============================================================================
# 8. Hold-out test
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli(C.HOLDOUT_STAGE)
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
spec = C.load_json(C.HOLDOUT_SPEC)
assert C.fingerprint_files(C.STIM_LOCAL, hm.path.tolist()) == spec["fingerprint"], "Hold-out files changed since locking."
d = hm[(hm.battery == "configural_tex") & hm.condition.isin(["cutG_intact", "scrambled"])].copy()
d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_STRENGTH"]))]
d = d[~d.category.isin(CONFIG["EXCLUDED"])]
keep = sorted(set(d[d.condition == "cutG_intact"].source_id) & set(d[d.condition == "scrambled"].source_id))
d = d[d.source_id.isin(keep)]
n_per = d.drop_duplicates("source_id").category.value_counts()
d = d[~d.category.isin(n_per[n_per < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
d = d.sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
OBJ = sorted(d.source_id.unique())
oi = {o: i for i, o in enumerate(OBJ)}
obj = d.source_id.map(oi).to_numpy()
is_grid = (d.condition == "cutG_intact").to_numpy()
yh = d.category.map(C.CAT_TO_IDX).to_numpy()
obj_cat = d.drop_duplicates("source_id").set_index("source_id").loc[OBJ, "category"].to_numpy()
grid_of_obj = np.array([np.where(is_grid & (obj == i))[0][0] for i in range(len(OBJ))])
SCR = np.zeros((len(d), len(OBJ)), np.float32)
for i in range(len(OBJ)):
    ix_ = np.where(~is_grid & (obj == i))[0]
    SCR[ix_, i] = 1.0 / len(ix_)
size = 64 if SMOKE else C.IMG_SIZE
HIMGS = torch.from_numpy(np.stack([np.asarray(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)))
                                   for p in d.path])).permute(0, 3, 1, 2).contiguous()
print(f"\n[3] hold-out: {len(OBJ)} objects, {len(d)} images")

rng = np.random.default_rng(C.derive_seed("stage12_holdout"))
w0 = np.zeros(len(OBJ), np.float32)
WB = np.zeros((CONFIG["N_BOOT"], len(OBJ)), np.float32)
cats = sorted(set(obj_cat))
for c in cats:
    g = np.where(obj_cat == c)[0]
    s_ = 1.0 / (len(g) * len(cats))
    w0[g] = s_
    draws = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
    cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
    np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), draws.ravel()), 1)
    WB[:, g] = cnt * s_


@torch.no_grad()
def decide(net, site, conds, mean_act):
    out = np.zeros((len(conds), len(d)), np.int8)
    mv = torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
    B = CONFIG["BATCH"]
    for b0 in range(0, len(d), B):
        a = net.prefix(norm(HIMGS[b0:b0 + B]), site)
        nb = a.shape[0]
        K = int(max(1, min(CONFIG["MAX_SUFFIX_IMAGES"] // nb, CONFIG["MAX_ELEMENTS"] // a.numel())))
        for s0 in range(0, len(conds), K):
            chunk = conds[s0:s0 + K]
            aa = a.unsqueeze(0).repeat(len(chunk), 1, 1, 1, 1)
            for k, (ch, mode) in enumerate(chunk):
                if len(ch) == 0:
                    continue
                chi = torch.as_tensor(np.asarray(ch, dtype=np.int64), device=DEVICE)
                if mode == "zero":
                    aa[k][:, chi] = 0
                else:
                    aa[k][:, chi] = mv[chi].to(aa.dtype).view(1, -1, 1, 1)
            pred = p16(net.suffix(aa.flatten(0, 1), site)).argmax(1).view(len(chunk), nb)
            out[s0:s0 + len(chunk), b0:b0 + nb] = pred.cpu().numpy()
    return out


def summ(point, boot):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return dict(estimate=float(point), lo=np.nan, hi=np.nan, p_two_sided=np.nan)
    lo, hi = np.quantile(b, [0.025, 0.975])
    p = min(1.0, 2 * min((1 + (b <= 0).sum()) / (len(b) + 1), (1 + (b >= 0).sum()) / (len(b) + 1)))
    return dict(estimate=float(point), lo=float(lo), hi=float(hi), p_two_sided=float(p))


def crr(ag, asc):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ag - asc) / (ag - C.CHANCE_16)


ROWS, LEVELS = [], []
for sname, S in OUT["sets"].items():
    labels = ["intact", "configural", "local"] + [f"size {i + 1}" for i in range(len(S["random_size_matched"]))] \
             + [f"damage {i + 1}" for i in range(len(S["random_damage_matched"]))]
    sets = [[], S["configural"], S["local"]] + S["random_size_matched"] + S["random_damage_matched"]
    for mode in ("zero", "mean"):
        ck = CKPT / f"holdout_{sname.replace(' ', '_')}_{mode}{sfx}.npz"
        if ck.exists():
            pred = np.load(ck)["pred"]
        else:
            if net is None:
                net = Split(MODEL)
            C.check_holdout_access(d.path.tolist(), MODEL)          # logs the run
            t0 = time.time()
            pred = decide(net, PRIMARY, [(s_, mode) for s_ in sets], OUT["mean_activation"])
            np.savez_compressed(ck, pred=pred, labels=np.array(labels, dtype=object))
            print(f"  {sname}, {mode} ablation: {len(sets)} conditions in {(time.time() - t0) / 60:0.1f} min")
        corr = (pred == yh[None, :]).astype(np.float32)
        G, Sc = corr[:, grid_of_obj].T, (corr @ SCR).T
        ag, asc, AG, AS = w0 @ G, w0 @ Sc, WB @ G, WB @ Sc
        ag_lo = np.quantile(AG, 0.025, axis=0)
        floor = ag_lo < CONFIG["FLOOR"]
        crp, CRB = crr(ag, asc), crr(AG, AS)
        loss, LOSS = ag - asc, AG - AS
        for i, l in enumerate(labels[:3]):
            LEVELS.append(dict(sets=sname, ablation=mode, condition=l, A_grid=float(ag[i]), A_grid_lo=float(ag_lo[i]),
                               A_scr=float(asc[i]), CR=float(crp[i]) if not floor[i] else np.nan, floor=bool(floor[i]),
                               loss=float(loss[i])))
        for kind in ("size", "damage"):
            ix = [i for i, l in enumerate(labels) if l.startswith(kind)]
            LEVELS.append(dict(sets=sname, ablation=mode, condition=f"random {kind}-matched (mean)", A_grid=float(ag[ix].mean()),
                               A_scr=float(asc[ix].mean()), CR=float(np.nanmean(np.where(floor[ix], np.nan, crp[ix]))),
                               loss=float(loss[ix].mean()), n_at_floor=int(floor[ix].sum()), n_sets=len(ix)))
        base = dict(sets=sname, ablation=mode)
        if floor[0] or floor[1]:
            ROWS.append(dict(**base, prediction="T1/T2", contrast="configural lesion at floor: not estimable", held=False))
        else:
            dC, DC = crp[1] - crp[0], CRB[:, 1] - CRB[:, 0]
            for kind in ("size", "damage"):
                ix = [i for i, l in enumerate(labels) if l.startswith(kind) and not floor[i]]
                if not ix:
                    ROWS.append(dict(**base, prediction="T1", contrast=f"all {kind}-matched sets at floor", held=False))
                    continue
                s_ = summ(dC - np.mean(crp[ix] - crp[0]), DC - np.mean(CRB[:, ix] - CRB[:, [0]], axis=1))
                row = dict(**base, prediction="T1", contrast=f"dCR configural minus mean dCR {kind}-matched random", **s_,
                           n_random_used=len(ix), n_random=len([l for l in labels if l.startswith(kind)]))
                if kind == "damage":
                    p5 = float(np.quantile(crp[ix] - crp[0], 0.05))
                    row.update(random_5th_percentile=p5, below_5th=bool(dC < p5))
                row["held"] = bool(s_["hi"] < 0 and row.get("below_5th", True))
                ROWS.append(row)
            if floor[2]:
                ROWS.append(dict(**base, prediction="T2", contrast="local lesion at floor", held=False))
            else:
                s_ = summ(crp[1] - crp[2], CRB[:, 1] - CRB[:, 2])
                ROWS.append(dict(**base, prediction="T2", contrast="dCR configural minus dCR local", **s_, held=bool(s_["hi"] < 0)))
        for kind in ("size", "damage"):
            ix = [i for i, l in enumerate(labels) if l.startswith(kind)]
            s_ = summ(loss[1] - loss[0] - np.mean(loss[ix] - loss[0]),
                      (LOSS[:, 1] - LOSS[:, 0]) - np.mean(LOSS[:, ix] - LOSS[:, [0]], axis=1))
            ROWS.append(dict(**base, prediction="T1 (absolute loss)", contrast=f"loss change configural minus {kind}-matched random",
                             **s_, held=bool(s_["hi"] < 0)))
        s_ = summ(loss[1] - loss[2], LOSS[:, 1] - LOSS[:, 2])
        ROWS.append(dict(**base, prediction="T2 (absolute loss)", contrast="loss configural minus loss local", **s_,
                         held=bool(s_["hi"] < 0)))

# =============================================================================
# 9. Save and summarise
# =============================================================================
pd.DataFrame(ROWS).to_csv(RES / f"contrasts{sfx}.csv", index=False)
pd.DataFrame(LEVELS).to_csv(RES / f"levels{sfx}.csv", index=False)
C.save_json(dict(config=CONFIG, sets_sha256=out_sha, primary_site=PRIMARY, sites=OUT["sites"],
                 n_planned=len(OUT["sets"]["planned"]["configural"]), n_matched=len(OUT["sets"]["matched cost"]["configural"]),
                 damage=dict(planned=OUT["sets"]["planned"]["damage_configural"],
                             matched=OUT["sets"]["matched cost"]["damage_configural"], target=TARGET),
                 run_at=time.strftime("%Y-%m-%d %H:%M:%S")), RES / f"stage12_summary{sfx}.json")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\nSites:")
print(pd.DataFrame(OUT["sites"]).T.to_string())
print("\nLevels:")
print(pd.DataFrame(LEVELS).round(3).to_string(index=False))
print("\nPredictions (exploratory):")
print(pd.DataFrame(ROWS).round(3).to_string(index=False))
if IN_COLAB:
    zp = Path("/content/stage12_results")
    shutil.make_archive(str(zp), "zip", root_dir=str(RES.parent), base_dir=RES.name)
    print(f"\nResults zipped to {zp}.zip (download it and send it over).")
print(f"\nSaved to {RES}")
