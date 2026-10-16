"""
STAGE 9: Final statistics, tables, figures and the post-freeze deviations log
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (no GPU needed; about 5 to 10 minutes).
No model is run and no image is loaded: everything is computed from the saved outputs of the earlier
stages (results/stage5, stage6, stage6b, stage8a, stage8b). Missing inputs are reported and that part is skipped.

What it produces
  1. plan/deviations_after_freeze.md: the clarifications that were written into the confirmatory scripts
     before they were run (hold-out lesion tests and agnosia signature tests), for the plan's deviations
     log and the repository. Exploratory additions are listed separately (Methods, not the log).
  2. Table 1: every hypothesis (H1 to H7), zero ablation, with 95% intervals, Holm-adjusted p within its
     family, the decision and the mean-ablation robustness verdict.
  3. Table 2: configural reliance of all twelve models on the hold-out set (descriptive).
  4. Table 3: retraining after progressive lesions (five training seeds). The interval combines both sources of
     variance: seed-to-seed (training) variance of the seed mean plus test-set variance (category-stratified
     object bootstrap of the seed mean), with a t quantile on n - 1 = 4 degrees of freedom. A two-level
     bootstrap (seeds and objects resampled) and a seed-only t interval are kept in the CSV for comparison;
     with five seeds the two-level bootstrap is too narrow, because resampling five values understates their spread.
  5. Figure 1: grid-control and scrambled accuracy after the primary-site lesions (zero ablation), with
     category-stratified object bootstrap intervals; random lesions as mean and 95% range of the sets.
  6. Figure 2: dose-response at the primary site (configural channels versus size-matched random sets).
  7. Figure 3: readout versus representation: (a) network and probe accuracy on grid controls, intact
     and after the configural lesion under zero and mean ablation; (b, c) configural reliance after
     readout-only and downstream retraining, seed means with two-level bootstrap intervals.

  8. A zip of every result file for writing (tables, summaries, figure images, plan records), saved in the
     Colab files panel (/content/agnosia_results_<date>.zip) with a copy in the project folder on Drive.

Outputs: results/stage9/*.csv|tex|json, figures/stage9/*.pdf|png, plan/deviations_after_freeze.md, the zip
"""

import os
import sys
import json
import time
import glob
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
from matplotlib.ticker import FuncFormatter

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
SFX = {k: os.environ.get(f"AGNOSIA_SFX_{k.upper()}", "_smoke" if SMOKE else "") for k in
       ["stage5", "stage6", "stage6b", "stage8a", "stage8b"]}
FROZEN = C.load_json(C.FROZEN_PLAN)
PLAN = FROZEN["plan"]
CONFIG = dict(MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]], MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
              EXCLUDED=PLAN["data"]["holdout_categories_excluded"], MIN_OBJECTS_PER_CATEGORY=2 if SMOKE else 20,
              N_BOOT=300 if SMOKE else 10000, SEED_T=0.975)
SETS = C.load_json(C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{'_smoke' if SMOKE else ''}.json")
RES = C.RESULTS_DIR / "stage9"
FIGS = C.FIG_DIR / "stage9"
for d_ in [RES, FIGS]:
    d_.mkdir(parents=True, exist_ok=True)
R = C.RESULTS_DIR
LABEL = {m: C.MODEL_INFO[m][0] for m in C.MODEL_INFO}
print(f"Stage 9 | plan {FROZEN['plan_sha256'][:12]} | reads saved outputs only")


def read(path):
    if Path(path).exists():
        return pd.read_csv(path)
    print(f"  missing: {path}")
    return None


def tidy(ax, axis, dec):
    f = FuncFormatter(lambda v, _: f"{0.0 if abs(v) < 1e-9 else v:0.{dec}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def nice(lo, hi, steps=(0.05, 0.1, 0.2, 0.25, 0.5, 1.0)):
    span = max(hi - lo, 1e-9)
    step = next((s for s in steps if span / s <= 6), steps[-1])
    return np.floor(lo / step) * step, np.ceil(hi / step) * step, step


def strat_weights(strata, key):
    rng = np.random.default_rng(C.derive_seed("stage9_boot", key))
    u = sorted(set(strata))
    W = np.zeros((CONFIG["N_BOOT"], len(strata)), np.float32)
    w0 = np.zeros(len(strata), np.float32)
    for c in u:
        g = np.where(strata == c)[0]
        s = 1.0 / (len(g) * len(u))
        w0[g] = s
        dr = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
        cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
        np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), dr.ravel()), 1)
        W[:, g] = cnt * s
    return w0, W


def cr(ag, asc):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ag - asc) / (ag - C.CHANCE_16)


# =============================================================================
# 1. Deviations log (clarifications fixed before the confirmatory runs)
# =============================================================================
DEVIATIONS = [
    ("Hold-out lesion tests (H1 to H3)", "Decision rule",
     "A hypothesis is supported when its interval rule is met and its Holm-adjusted two-sided bootstrap p is below 0.05 "
     "(the plan lists both without stating how they combine). H2 has two contrasts (size- and damage-matched controls); "
     "its p-value is the larger of the two (intersection-union test)."),
    ("Hold-out lesion tests (H1 to H3)", "Random control lesions at the floor",
     "A random control set whose own grid-control accuracy is at the floor has no CR; it is left out of the random mean "
     "and of the 5th-percentile test, and the number left out is reported. A sensitivity analysis keeps those sets "
     "(CR computed whenever grid-control accuracy exceeds chance)."),
    ("Agnosia signature tests (H4 to H7)", "Hue labels",
     "The dominant hue of an object is taken from its grid-control image (6 bins of 60 degrees over pixels with "
     "saturation above 0.2); objects with fewer than 100 saturated pixels have no hue label and are left out of the hue probe."),
    ("Agnosia signature tests (H4 to H7)", "Control tasks",
     "Hold-out objects are new, so a probe trained on labels shuffled at object level is at chance on them by construction. "
     "Selectivity is therefore measured on the calibration set by holding out one scramble variant per object, for the two "
     "scrambled-image probes; grid-control and multi-label probes have one image per object, so no control task is defined for them."),
    ("Agnosia signature tests (H4 to H7)", "Resampling units and H4 p-value",
     "Bootstrap units: hold-out objects stratified by category (H4, H5), the 160 reference image pairs stratified by category "
     "(H6), hold-out figure pairs stratified by the first object's category (H7). Probes are fitted once per condition, so "
     "intervals cover test-set sampling. H4's p-value is computed against the 0.05 margin; with two probes, the larger p is used."),
]
EXPLORATORY = [
    "Unit-level piece interactions (exact Walsh decomposition over piece subsets) and the split of the scramble effect into "
    "position-coded parts and interactions, calibration test half.",
    "Concentration of network decisions and the readout gap (probe minus network accuracy), from the saved hold-out decisions.",
    "Progressive lesioning with brief retraining (named in the plan as exploratory), with a numerical safeguard added after "
    "one track diverged, and its five-seed replication with a log-space loss.",
    "Navon letter probes, Frankenstein effects after lesions, per-category results and the twelve-model profile "
    "(all named in the plan as exploratory).",
]
md = ["# Deviations after the freeze: clarifications to the confirmatory analyses", "",
      f"Plan SHA-256: `{FROZEN['plan_sha256']}`. Amendment A1 SHA-256: "
      "`6dd4a7445862ba90f392a6028623abdf4d98340ada89fa8ff6f396bc5b814e9f`.", "",
      "Each clarification below was written into the header of the analysis script before that script was first run on "
      "the hold-out set. None changes a hypothesis, measure, unit set or threshold.", "",
      "| Analysis | Point | Clarification |", "| --- | --- | --- |"]
md += [f"| {a} | {p} | {t} |" for a, p, t in DEVIATIONS]
md += ["", "## Exploratory analyses added after the freeze (reported in the Methods, no confirmatory claim)", ""]
md += [f"- {x}" for x in EXPLORATORY]
dev_path = C.PLAN_DIR / f"deviations_after_freeze{'_smoke' if SMOKE else ''}.md"
dev_path.write_text("\n".join(md) + "\n")
(RES / "deviations_after_freeze.md").write_text("\n".join(md) + "\n")
print(f"  deviations log written: {dev_path}")

# =============================================================================
# 2. Table 1: hypotheses H1 to H7
# =============================================================================
h5c = read(R / "stage5" / f"hypothesis_contrasts{SFX['stage5']}.csv")
h5f = read(R / "stage5" / f"primary_family_holm{SFX['stage5']}.csv")
h5r = read(R / "stage5" / f"robustness_verdicts{SFX['stage5']}.csv")
h6c = read(R / "stage6" / f"hypothesis_contrasts{SFX['stage6']}.csv")
h6f = read(R / "stage6" / f"secondary_family_holm{SFX['stage6']}.csv")
t1 = []
if h5c is not None and h5f is not None:
    for r in h5c.itertuples():
        f = h5f[(h5f.model == r.model) & (h5f.hypothesis == r.hypothesis)].iloc[0]
        rob = "" if h5r is None else h5r[(h5r.model == r.model) & (h5r.hypothesis == r.hypothesis)]
        t1.append(dict(family="primary", model=r.model, hypothesis=r.hypothesis, contrast=r.contrast,
                       estimate=r.estimate, lo=r.lo, hi=r.hi, floor=bool(r.floor), p_holm=f.p_holm, supported=bool(f.supported),
                       robust=(bool(rob.robust.iloc[0]) if isinstance(rob, pd.DataFrame) and len(rob) else
                               (bool(f.supported) if r.hypothesis == "H1" else False))))
if h6c is not None and h6f is not None:
    names = {"H4 hue": "hue probe drop, configural minus random", "H4 catscr": "scrambled-category probe drop, configural minus random",
             "H5": "change in error distance to BagNet-33, configural minus random",
             "H6": "change in edge minus silhouette accuracy, configural minus random",
             "H7": "change in overlap cost, configural minus random"}
    for r in h6c[h6c.ablation == "zero"].itertuples():
        f = h6f[(h6f.model == r.model) & (h6f.hypothesis == r.hypothesis)].iloc[0]
        t1.append(dict(family="secondary", model=r.model, hypothesis=r.hypothesis, contrast=names.get(r.contrast, r.contrast),
                       estimate=r.estimate, lo=r.lo, hi=r.hi, floor=False, p_holm=f.p_holm, supported=bool(f.supported),
                       robust=bool(f.robust)))
t1 = pd.DataFrame(t1)
if len(t1):
    t1.to_csv(RES / "table1_hypotheses.csv", index=False)
    L = [r"\begin{tabular}{llp{6.2cm}rrll}", r"\hline",
         r"Model & H & Contrast & Estimate [95\% CI] & $p_\mathrm{Holm}$ & Supported & Robust \\", r"\hline"]
    for r in t1.itertuples():
        est = "floor" if r.floor else f"{r.estimate:0.3f} [{r.lo:0.3f}, {r.hi:0.3f}]"
        L.append(f"{LABEL[r.model]} & {r.hypothesis} & {r.contrast.replace('dCR', r'$\Delta$CR')} & {est} & "
                 f"{r.p_holm:0.4f} & {'yes' if r.supported else 'no'} & {'yes' if r.robust else 'no'} \\\\")
    L += [r"\hline", r"\end{tabular}"]
    (RES / "table1_hypotheses.tex").write_text("\n".join(L))

# =============================================================================
# 3. Table 2: model profile
# =============================================================================
prof = read(R / "stage8b" / f"model_profile{SFX['stage8b']}.csv")
if prof is not None:
    prof.to_csv(RES / "table2_model_profile.csv", index=False)
    L = [r"\begin{tabular}{lllrrr}", r"\hline", r"Model & Family & Role & Grid acc. & Scrambled acc. & CR [95\% CI] \\", r"\hline"]
    for r in prof.itertuples():
        crs = "floor" if not np.isfinite(r.CR) else f"{r.CR:0.3f} [{r.CR_lo:0.3f}, {r.CR_hi:0.3f}]"
        L.append(f"{r.label} & {r.family} & {r.role} & {r.A_grid:0.3f} & {r.A_scr:0.3f} & {crs} \\\\")
    L += [r"\hline", r"\end{tabular}"]
    (RES / "table2_model_profile.tex").write_text("\n".join(L))

# =============================================================================
# 4. Hold-out bookkeeping (manifest only)
# =============================================================================
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)


def holdout_table(conditions):
    d = hm[(hm.battery == "configural_tex") & hm.condition.isin(conditions)].copy()
    d = d[~((d.condition == "scrambled") & (d.manip_strength < CONFIG["MIN_STRENGTH"]))]
    d = d[~d.category.isin(CONFIG["EXCLUDED"])]
    keep = None
    for c in conditions:
        s = set(d[d.condition == c].source_id)
        keep = s if keep is None else keep & s
    d = d[d.source_id.isin(keep)]
    n = d.drop_duplicates("source_id").category.value_counts()
    d = d[~d.category.isin(n[n < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
    d = d.sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)
    objs = sorted(d.source_id.unique())
    oi = {o: i for i, o in enumerate(objs)}
    ob = d.source_id.map(oi).to_numpy()
    cat = d.drop_duplicates("source_id").set_index("source_id").loc[objs, "category"].to_numpy()
    grid = np.array([np.where((d.condition == "cutG_intact").to_numpy() & (ob == i))[0][0] for i in range(len(objs))])
    S = np.zeros((len(d), len(objs)), np.float32)
    for i in range(len(objs)):
        idx = np.where((d.condition == "scrambled").to_numpy() & (ob == i))[0]
        S[idx, i] = 1.0 / len(idx)
    return d, cat, grid, S


# =============================================================================
# 5. Figure 1: accuracies after the primary-site lesions (zero ablation)
# =============================================================================
d5, CAT5, G5, S5 = holdout_table(["cutG_intact", "scrambled"])
W5 = strat_weights(CAT5, "fig1")
fig1 = []
for name in CONFIG["MODELS"]:
    f = R / "stage5" / "checkpoints" / f"{name}_primary_zero{SFX['stage5']}.npz"
    if not f.exists():
        print(f"  missing: {f}")
        continue
    z = np.load(f, allow_pickle=True)
    labels, corr = list(z["labels"]), z["correct"]
    if corr.shape[1] != len(d5):
        print(f"  {f.name} does not match the hold-out list; Figure 1 skipped for {name}")
        continue
    w0, W = W5
    for lab in ["intact", "configural", "local"]:
        c_ = corr[labels.index(lab)]
        g, s = c_[G5].astype(np.float32), c_.astype(np.float32) @ S5
        for meas, v in [("grid", g), ("scrambled", s)]:
            b = W @ v
            fig1.append(dict(model=name, condition=lab, measure=meas, value=float(w0 @ v), lo=float(np.quantile(b, 0.025)),
                             hi=float(np.quantile(b, 0.975))))
    dm = [l for l in labels if l.startswith("damage ")]
    for meas in ["grid", "scrambled"]:
        vals = []
        for l in dm:
            c_ = corr[labels.index(l)]
            v = c_[G5].astype(np.float32) if meas == "grid" else c_.astype(np.float32) @ S5
            vals.append(float(w0 @ v))
        vals = np.array(vals)
        fig1.append(dict(model=name, condition="random (damage-matched)", measure=meas, value=float(vals.mean()),
                         lo=float(np.quantile(vals, 0.025)), hi=float(np.quantile(vals, 0.975))))
fig1 = pd.DataFrame(fig1)
fig1.to_csv(RES / "figure1_accuracies.csv", index=False)


def figure1():
    if fig1.empty:
        return
    C.set_pub_style()
    conds = ["intact", "configural", "local", "random (damage-matched)"]
    xl = ["Intact", "Configural\nlesion", "Local\nlesion", "Random\nlesions"]
    fig, axes = plt.subplots(1, len(CONFIG["MODELS"]), figsize=(8.5 * len(CONFIG["MODELS"]), 6.8))
    hi_all = max(0.1, fig1.hi.max())
    top = np.ceil(hi_all / 0.1) * 0.1
    for p, (ax, name) in enumerate(zip(axes, CONFIG["MODELS"])):
        g = fig1[fig1.model == name]
        for meas, mk, mfc, dx, lab in [("grid", "o", "#000000", -0.12, "Grid control"), ("scrambled", "o", "white", 0.12, "Scrambled")]:
            r = g[g.measure == meas].set_index("condition").reindex(conds)
            x = np.arange(len(conds)) + dx
            ax.errorbar(x, r.value, yerr=[r.value - r.lo, r.hi - r.value], fmt=mk, ms=10, capsize=5, lw=1.8, color="#000000",
                        mfc=mfc, mec="#000000", mew=2, label=lab if p == 0 else None)
        ax.axhline(C.CHANCE_16, color="#8a8a8a", ls=":", lw=1.5, label="Chance (1/16)" if p == 0 else None)
        ax.axhline(2 * C.CHANCE_16, color="#8a8a8a", ls="--", lw=1.5, label="Floor (2/16)" if p == 0 else None)
        C.style_axes(ax, ylim=(0, top), ystep=0.1, decimals=1)
        tidy(ax, "y", 1)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(xl)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_ylabel(f"Accuracy (16-way)\n{LABEL[name]}, {SETS['models'][name]['primary_site']}")
        ax.yaxis.set_label_coords(-0.13, 0.5)
        ax.text(-0.13, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.3, bottom=0.24)
    C.save_figure(fig, FIGS / "figure1_lesion_accuracies")
    plt.close(fig)


figure1()

# =============================================================================
# 6. Figure 2: dose-response
# =============================================================================
dose = read(R / "stage5" / f"dose_response{SFX['stage5']}.csv")


def figure2():
    if dose is None:
        return
    C.set_pub_style()
    fig, axes = plt.subplots(1, len(CONFIG["MODELS"]), figsize=(8.5 * len(CONFIG["MODELS"]), 6.8))
    v = pd.concat([dose.dCR_random_min, dose.dCR_random_max, dose.get("dCR_configural_lo"), dose.get("dCR_configural_hi")]).dropna()
    lo, hi, step = nice(min(0.0, v.min()), max(0.0, v.max()))
    for p, (ax, name) in enumerate(zip(axes, CONFIG["MODELS"])):
        dd = dose[dose.model == name].sort_values("fraction")
        x = 100 * dd.fraction.to_numpy()
        ok = np.isfinite(dd.dCR_random_mean.to_numpy())
        ax.fill_between(x[ok], dd.dCR_random_min[ok], dd.dCR_random_max[ok], color="#8a8a8a", alpha=0.3, lw=0,
                        label="Random sets, range" if p == 0 else None)
        ax.plot(x[ok], dd.dCR_random_mean[ok], color="#8a8a8a", marker="D", ms=8, lw=2, label="Random sets, mean" if p == 0 else None)
        okc = ~dd.configural_floor.astype(bool).to_numpy()
        yv = dd.dCR_configural.to_numpy()
        ax.errorbar(x[okc], yv[okc], yerr=[yv[okc] - dd.dCR_configural_lo.to_numpy()[okc], dd.dCR_configural_hi.to_numpy()[okc] - yv[okc]],
                    color="#2a78d6", marker="o", ms=9, lw=2, capsize=5, label="Configural channels (by score)" if p == 0 else None)
        for xf in x[~okc]:
            ax.annotate("floor", (xf, 0), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=14, color="#2a78d6")
        ax.axhline(0, color="#000000", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 50)
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
        ax.minorticks_off()
        dec = 1 if round(step * 10, 6) == int(round(step * 10, 6)) else 2      # 0.25 or 0.05 need two decimals
        C.style_axes(ax, ylim=(lo, hi), ystep=step, decimals=dec)
        tidy(ax, "y", dec)
        ax.set_xlabel(f"Channels lesioned at {SETS['models'][name]['primary_site']} (%)")
        ax.set_ylabel(f"Change in configural reliance\n{LABEL[name]}")
        ax.yaxis.set_label_coords(-0.13, 0.5)
        ax.text(-0.13, 1.03, "ab"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.3, bottom=0.22)
    C.save_figure(fig, FIGS / "figure2_dose_response")
    plt.close(fig)


figure2()

# =============================================================================
# 7. Table 3 and Figure 3: retraining (five seeds, two-level bootstrap)
# =============================================================================
d8, CAT8, G8, S8 = holdout_table(["cutG_intact", "scrambled", "intact"])
w08, W8 = strat_weights(CAT8, "seeds")
seed_rng = np.random.default_rng(C.derive_seed("stage9_seeds"))
rows8 = {}
for f in sorted(glob.glob(str(R / "stage8a" / "checkpoints" / f"*_seed*_????????{SFX['stage8a']}.npz"))):
    z = np.load(f, allow_pickle=True)
    for r in z["rows"]:
        if r["phase"] != "after":
            continue
        rows8[(r["model"], r["regime"], r["order"], r["seed"], r["step"])] = r
def combined_interval(P, B):
    """95% interval for a seed mean with both variance sources: seed-to-seed (training) variance over n seeds and
    test-set variance (object bootstrap of the seed mean, seeds fixed); t quantile with n - 1 degrees of freedom."""
    n = len(P)
    if n < 2:
        return np.nan, np.nan
    se_seed2 = P.var(ddof=1) / n
    se_obj2 = np.nanvar(np.nanmean(B, 0), ddof=1)
    half = stats.t.ppf(CONFIG["SEED_T"], n - 1) * np.sqrt(se_seed2 + se_obj2)
    return float(P.mean() - half), float(P.mean() + half)


t3, lev = [], []
if rows8 and next(iter(rows8.values()))["correct"].shape[0] != len(d8):
    print("  retraining checkpoints do not match the hold-out list; Table 3 skipped")
    rows8 = {}
for name in CONFIG["MODELS"]:
    keys = [k for k in rows8 if k[0] == name]
    if not keys:
        continue
    last = max(k[4] for k in keys)
    seeds = sorted({k[3] for k in keys})
    for regime in ["readout", "downstream"]:
        per = {}
        for seed in seeds:
            tracks = {k[2]: rows8[k] for k in keys if k[1] == regime and k[3] == seed and k[4] == last}
            if any(t["train_acc"] < 0.5 for t in tracks.values()):
                tracks = {o: t for o, t in tracks.items() if t["train_acc"] >= 0.5}
            vals = {}
            for o, t in tracks.items():
                c_ = t["correct"]
                g, s = c_[G8].astype(np.float32), c_.astype(np.float32) @ S8
                vals[o] = dict(point=float(cr(w08 @ g, w08 @ s)), boot=cr(W8 @ g, W8 @ s),
                               gpoint=float(w08 @ g), gboot=W8 @ g)
            per[seed] = vals
        groups = {"configural": lambda v: v.get("configural"), "no lesion": lambda v: v.get("none")}
        for seed, v in per.items():
            rk = [o for o in v if o.startswith("random")]
            if rk:
                v["random mean"] = dict(point=float(np.mean([v[o]["point"] for o in rk])), boot=np.mean([v[o]["boot"] for o in rk], 0),
                                        gpoint=float(np.mean([v[o]["gpoint"] for o in rk])), gboot=np.mean([v[o]["gboot"] for o in rk], 0))
        valid = [s for s in seeds if "configural" in per[s] and "none" in per[s] and "random mean" in per[s]]
        nb = CONFIG["N_BOOT"]
        draws = seed_rng.integers(0, len(valid), size=(nb, len(valid)))
        for lab, key in [("configural", "configural"), ("no lesion", "none"), ("random (mean of 3 orders)", "random mean")]:
            P = np.array([per[s][key]["point"] for s in valid])
            B = np.stack([per[s][key]["boot"] for s in valid], 0)
            lo_c, hi_c = combined_interval(P, B)
            lev.append(dict(model=name, regime=regime, lesion=lab, n_seeds=len(valid), CR=float(P.mean()), lo=lo_c, hi=hi_c))
        for ref_lab, ref in [("no lesion", "none"), ("random (mean of 3 orders)", "random mean")]:
            for meas, pk, bk in [("CR", "point", "boot"), ("grid accuracy", "gpoint", "gboot")]:
                P = np.array([per[s]["configural"][pk] - per[s][ref][pk] for s in valid])
                B = np.stack([per[s]["configural"][bk] - per[s][ref][bk] for s in valid], 0)
                hb = B[draws, np.arange(nb)[:, None]].mean(1)
                se = P.std(ddof=1) / np.sqrt(len(P)) if len(P) > 1 else np.nan
                tq = stats.t.ppf(CONFIG["SEED_T"], len(P) - 1) if len(P) > 1 else np.nan
                lo_c, hi_c = combined_interval(P, B)
                t3.append(dict(model=name, regime=regime, contrast=f"configural minus {ref_lab}", measure=meas, n_seeds=len(P),
                               estimate=float(P.mean()), lo=lo_c, hi=hi_c, lo_two_level=float(np.nanquantile(hb, 0.025)),
                               hi_two_level=float(np.nanquantile(hb, 0.975)), lo_t=float(P.mean() - tq * se),
                               hi_t=float(P.mean() + tq * se), seed_min=float(P.min()), seed_max=float(P.max()),
                               seeds_same_sign=int((np.sign(P) == np.sign(P.mean())).sum())))
t3, lev = pd.DataFrame(t3), pd.DataFrame(lev)
if len(t3):
    t3.to_csv(RES / "table3_retraining.csv", index=False)
    lev.to_csv(RES / "figure3_retraining_levels.csv", index=False)
    L = [r"\begin{tabular}{lllrrl}", r"\hline",
         r"Model & Retraining & Contrast (CR) & Estimate [95\% CI] & Seed range & Same sign \\", r"\hline"]
    for r in t3[t3.measure == "CR"].itertuples():
        L.append(f"{LABEL[r.model]} & {r.regime} & {r.contrast} & {r.estimate:0.3f} [{r.lo:0.3f}, {r.hi:0.3f}] & "
                 f"{r.seed_min:0.3f} to {r.seed_max:0.3f} & {r.seeds_same_sign}/{r.n_seeds} \\\\")
    L += [r"\hline", r"\end{tabular}"]
    (RES / "table3_retraining.tex").write_text("\n".join(L))

rm = read(R / "stage6b" / f"readout_measures{SFX['stage6b']}.csv")


def figure3():
    if rm is None and lev.empty:
        return
    C.set_pub_style()
    n = 1 + len(CONFIG["MODELS"])
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 6.8))
    ax = axes[0]
    if rm is not None:
        sel = [("none", "intact", "Intact"), ("zero", "configural", "Configural,\nzero"), ("mean", "configural", "Configural,\nmean")]
        cols = {CONFIG["MODELS"][0]: "#2a78d6", CONFIG["MODELS"][1]: "#eb6834"}
        for i, name in enumerate(CONFIG["MODELS"]):
            for meas, mfc, lab in [("accuracy", "white", "network"), ("probe", None, "probe")]:
                ys, lo_, hi_ = [], [], []
                for abl, cond, _ in sel:
                    r = rm[(rm.model == name) & (rm.ablation == abl) & (rm.condition == cond) & (rm.measure == meas)]
                    ys.append(r.value.iloc[0] if len(r) else np.nan)
                    lo_.append(r.lo.iloc[0] if len(r) else np.nan)
                    hi_.append(r.hi.iloc[0] if len(r) else np.nan)
                ys, lo_, hi_ = map(np.array, (ys, lo_, hi_))
                x = np.arange(len(sel)) + (i - 0.5) * 0.3
                ax.errorbar(x, ys, yerr=[ys - lo_, hi_ - ys], fmt="o" if meas == "probe" else "s", ms=9, capsize=4, lw=1.6,
                            color=cols[name], mfc=mfc or cols[name], mew=2, label=f"{LABEL[name]}, {lab}")
        C.style_axes(ax, ylim=(0, 1), ystep=0.2, decimals=1)
        tidy(ax, "y", 1)
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels([s[2] for s in sel])
        ax.set_xlim(-0.6, len(sel) - 0.4)
        ax.set_ylabel("Grid-control accuracy")
    if not lev.empty:
        lv = lev.CR.tolist() + lev.lo.tolist() + lev.hi.tolist()
        lo, hi, step = nice(min(0.0, np.nanmin(lv)), max(0.6, np.nanmax(lv)))
        mk = {"no lesion": ("o", "white", "#000000"), "configural": ("o", None, "#2a78d6"),
              "random (mean of 3 orders)": ("D", None, "#8a8a8a")}
        for p, name in enumerate(CONFIG["MODELS"]):
            ax = axes[1 + p]
            g = lev[lev.model == name]
            for j, (lesion, (m_, mfc, col)) in enumerate(mk.items()):
                r = g[g.lesion == lesion].set_index("regime").reindex(["readout", "downstream"])
                x = np.arange(2) + (j - 1) * 0.22
                ax.errorbar(x, r.CR, yerr=[r.CR - r.lo, r.hi - r.CR], fmt=m_, ms=10, capsize=5, lw=1.8, color=col,
                            mfc=mfc or col, mew=2, label=lesion.capitalize() if p == 0 else None)
            dec = 1 if round(step * 10, 6) == int(round(step * 10, 6)) else 2
            C.style_axes(ax, ylim=(lo, hi), ystep=step, decimals=dec)
            tidy(ax, "y", dec)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Readout\nretrained", "Downstream\nretrained"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylabel(f"Configural reliance after retraining\n{LABEL[name]}")
    for p, ax in enumerate(axes):
        ax.yaxis.set_label_coords(-0.16, 0.5)
        ax.text(-0.16, 1.03, "abc"[p], transform=ax.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    if len(axes) > 1 and not lev.empty:
        h2, l2 = axes[1].get_legend_handles_labels()
        h, l = h + h2, l + l2
    leg = fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    C.outline_legend(leg)
    fig.subplots_adjust(wspace=0.45, bottom=0.26)
    C.save_figure(fig, FIGS / "figure3_readout_and_retraining")
    plt.close(fig)


figure3()

C.save_json(dict(plan_sha256=FROZEN["plan_sha256"], inputs=SFX, n_boot=CONFIG["N_BOOT"], run_at=time.strftime("%Y-%m-%d %H:%M:%S")),
            RES / "stage9_summary.json")


# =============================================================================
# 8. One zip of every result file for writing (tables, summaries, figures, plan records)
# =============================================================================
def bundle():
    """Collects csv, json, tex, md and txt results from every stage, figures as png, and the plan records
    (frozen plan, amendment, hold-out specification, access log, deviations). Model checkpoints (npz), stimuli
    and anything above 25 MB are left out. Written to /content (and a copy to the project folder on Drive)."""
    import zipfile
    keep_ext = {".csv", ".json", ".tex", ".md", ".txt", ".jsonl"}
    files = []
    for root, sub in [(C.RESULTS_DIR, "results"), (C.FIG_DIR, "figures"), (C.PLAN_DIR, "plan")]:
        for f in sorted(Path(root).rglob("*")):
            if not f.is_file() or "checkpoints" in f.parts:
                continue
            ext = f.suffix.lower()
            if sub == "figures" and ext != ".png":
                continue
            if sub != "figures" and ext not in keep_ext:
                continue
            files.append((f, Path(sub) / f.relative_to(root)))
    for name in ["stimuli/stage1_gate_report.json", "stimuli/holdout_manifest.csv", "stimuli/frozen_calib_config.json"]:
        f = C.DRIVE_ROOT / name
        if f.exists():
            files.append((f, Path(name)))
    out_dir = Path("/content") if IN_COLAB else RES
    stamp = time.strftime("%Y%m%d_%H%M")
    zpath = out_dir / f"agnosia_results_{stamp}.zip"
    skipped = []
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f, arc in files:
            if f.stat().st_size > 25e6:
                skipped.append(str(arc))
                continue
            z.write(f, arcname=str(Path("agnosia_results") / arc))
        z.writestr("agnosia_results/README.txt",
                   "Results of the CNN visual agnosia project (all stages), collected by the final-statistics script.\n"
                   f"Created {time.strftime('%Y-%m-%d %H:%M:%S')}. Plan SHA-256 {FROZEN['plan_sha256']}.\n"
                   "results/<stage>/: tables (csv, tex) and summaries (json); figures/<stage>/: png; plan/: frozen plan, "
                   "amendment, hold-out specification, access log, deviations.\n"
                   "Left out: model checkpoints (npz), stimuli and files above 25 MB"
                   + (f" ({', '.join(skipped)})" if skipped else "") + ".\n")
    mb = zpath.stat().st_size / 1e6
    if IN_COLAB:
        import shutil
        shutil.copy(zpath, C.DRIVE_ROOT / zpath.name)
    print(f"\n  results bundle: {zpath} ({len(files) - len(skipped)} files, {mb:0.1f} MB)"
          + (f"; copy on Drive: {C.DRIVE_ROOT / zpath.name}" if IN_COLAB else "")
          + (f"; skipped (too large): {skipped}" if skipped else ""))
    return zpath


ZIP_PATH = bundle()
pd.set_option("display.width", 240)
show = lambda df: df.round(4).astype(object).where(df.notna(), "").to_string(index=False)
print("\nTable 1. Hypotheses (zero ablation decides; robust = mean ablation agrees)")
print(show(t1) if len(t1) else "  (inputs missing)")
print("\nTable 3. Retraining after progressive lesions (lo/hi: seed and object variance combined, t with 4 df)")
print(show(t3) if len(t3) else "  (inputs missing)")
print("\nFigure 3 levels (CR after retraining, seed means)")
print(show(lev) if len(lev) else "  (inputs missing)")
print(f"\nDone. Outputs in {RES}, {FIGS} and {dev_path}")
print(f"Results zip to share: {ZIP_PATH} (Files panel on the left; right-click, Download)")
