"""
STAGE 11: ResNet-50 (SWSL) at a testable lesion size, and shape bias after lesions (exploratory)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 1 to 1.5 hours on an A100). Each part is
saved when it finishes; if the runtime disconnects, rerunning picks up where it stopped. At the end the
results folder is zipped to /content for download.

Why. A pre-submission review raised two gaps that this script addresses:
  (a) The planned configural set of ResNet-50 (SWSL) cost so much recognition that the selective-lesion
      tests (H2, H3) could not be computed in the primary model.
  (b) The local lesion is "local" only by how it was selected (its effect on scrambled images), so the
      opposite effects of the configural and local lesions are partly built in. An independent measure
      of local versus global processing is needed.
Everything here is exploratory, labelled as such in the paper, and carries no confirmatory claim. The
predictions below were written before this script was first run, and are reported whether they hold.

Part A. ResNet-50 (SWSL) at a lesion size it survives
  Selection uses the calibration localizer half only, with the published channel scores (I_CR, A1):
    target damage   the damage of the CORnet-S configural set (drop in grid-control accuracy on the
                    localizer half, stored in lesion_sets_A1.json), so that the two models are lesioned
                    at the same cost
    configural set  the top k layer3 channels by I_CR, with k (1 to 224) chosen so that the set's damage
                    is closest to the target (smallest k among ties)
    local set       the same rule as the planned local set: channels with I_CR at or below the site
                    median, ranked by local score, sized so that its damage is closest to the
                    configural set's
    random sets     100 size-matched sets; 100 damage-matched sets of the same size (pool of up to
                    10,000 random sets, damage within 0.02, else the 100 closest), as planned
  The sets are written to results/stage11/sets_resnet_small.json, with their SHA-256 printed, before any
  hold-out image is used. They are then tested on the hold-out set exactly as H2 and H3 were, under
  zero and mean ablation: configural reliance with the floor rule, random sets at floor left out and
  counted, category-stratified bootstrap over objects, plus the floor-free absolute scramble loss.
  R1  dCR(configural) minus mean dCR(random) has upper bound < 0 for size- and damage-matched sets,
      and dCR(configural) lies below the 5th percentile of the damage-matched sets.
  R2  dCR(configural) minus dCR(local) has upper bound < 0.
  Zero ablation is reported first, as in the plan; mean ablation as robustness.

Part B. Shape bias on cue-conflict images, an independent measure of local versus global processing
  The 1280 public cue-conflict images of the model-versus-human benchmark (shape of one category, texture
  of another); images whose shape and texture categories coincide are dropped. Shape bias = share of
  shape decisions among decisions that are either the shape or the texture category. None of these
  images was used to select any lesion set.
  Conditions per model, at the primary site: intact; the planned configural and local sets; the first
  20 damage-matched random sets; each lesion under zero and mean ablation. For ResNet-50 (SWSL) the
  small sets of Part A are added in the same way.
  Bootstrap over images stratified by shape category (10,000 resamples). A lesioned condition is
  estimable if at least 25% of images receive a shape or a texture decision.
  S1  The configural lesion lowers shape bias more than the random lesions (change minus mean random
      change, upper bound < 0).
  S2  The local lesion raises shape bias relative to the random lesions (lower bound > 0).
  S3  Change after the local lesion minus change after the configural lesion has lower bound > 0.
  If S1 and S2 hold, the two lesions differ on a measure that played no part in selecting them.

Outputs: results/stage11/*.csv|json, and /content/stage11_results.zip for download.
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
import torch
import torch.nn as nn
from PIL import Image

SMOKE = os.environ.get("AGNOSIA_SMOKE") is not None      # offline testing only
FAKE = SMOKE and os.environ.get("AGNOSIA_FAKE") is not None   # offline test of the statistics with synthetic decisions
sfx = "_smoke" if SMOKE else ""

PLAN_SHA_REGISTERED = "dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c"
SETS_SHA_REGISTERED = "9bb05eb1ee8729786246ff80f23575225c88f3271c4c6756a92525500c2fb904"

assert C.FROZEN_PLAN.exists(), "The analysis plan is not frozen."
FROZEN = C.load_json(C.FROZEN_PLAN)
PLAN = FROZEN["plan"]
ok, why = C.holdout_status()
assert ok, f"Hold-out locked: {why}"
SETS_PATH = C.RESULTS_DIR / "stage3" / f"lesion_sets_A1{sfx}.json"
SCORES_PATH = C.RESULTS_DIR / "stage3" / f"channel_scores_A1{sfx}.csv"
sets_sha = hashlib.sha256(SETS_PATH.read_bytes()).hexdigest()
if not SMOKE:
    assert FROZEN["plan_sha256"] == PLAN_SHA_REGISTERED, "Frozen plan differs from the registered plan."
    assert sets_sha == SETS_SHA_REGISTERED, "lesion_sets_A1.json differs from the published version."
SETS = C.load_json(SETS_PATH)
SCORES = pd.read_csv(SCORES_PATH)

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    RESNET="resnet50_swsl",
    CORNET="cornet_s",
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    FLOOR=2 * C.CHANCE_16,
    N_SIZE=100,
    N_DAMAGE=100,
    DAMAGE_POOL=10000,
    DAMAGE_TOL=0.02,
    N_RANDOM_CUE=20,
    MIN_DECIDED=0.25,
    N_BOOT=10000,
    BATCH=128,
    MAX_SUFFIX_IMAGES=1024,
    MAX_ELEMENTS=4e8,
)
if SMOKE:
    CONFIG.update(N_SIZE=5, N_DAMAGE=5, DAMAGE_POOL=40, N_RANDOM_CUE=3, N_BOOT=300, BATCH=32,
                  MIN_OBJECTS_PER_CATEGORY=2)

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True     # same numerical settings as the earlier lesion analyses
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage11"
CKPT = RES / "checkpoints"
for dd in [RES, CKPT]:
    dd.mkdir(parents=True, exist_ok=True)
print(f"Stage 11 (exploratory) | plan {FROZEN['plan_sha256'][:12]} | lesion sets {sets_sha[:12]} | device {DEVICE}")

size = 64 if SMOKE else C.IMG_SIZE
MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def load_imgs(paths):
    return torch.from_numpy(np.stack([np.asarray(Image.open(C.STIM_LOCAL / p).convert("RGB").resize((size, size)))
                                      for p in paths])).permute(0, 3, 1, 2).contiguous()


def norm(x_uint8):
    return (x_uint8.to(DEVICE, non_blocking=True).float() / 255.0 - MEAN) / STD


# =============================================================================
# 1. Images: calibration localizer half, hold-out, cue-conflict
# =============================================================================
if IN_COLAB:
    C.unpack_stimuli("frozen_calib")
    C.unpack_stimuli("stage1")
    C.unpack_stimuli(C.HOLDOUT_STAGE)


def configural_frame(man, split=None):
    t = man[(man.battery == "configural_tex") & man.condition.isin(["cutG_intact", "scrambled"])].copy()
    if split is not None:
        t = t[t.split == split]
    t = t[~((t.condition == "scrambled") & (t.manip_strength < CONFIG["MIN_STRENGTH"]))]
    t = t[~t.category.isin(CONFIG["EXCLUDED"])]
    keep = sorted(set(t[t.condition == "cutG_intact"].source_id) & set(t[t.condition == "scrambled"].source_id))
    t = t[t.source_id.isin(keep)]
    n_per = t.drop_duplicates("source_id").category.value_counts()
    t = t[~t.category.isin(n_per[n_per < CONFIG["MIN_OBJECTS_PER_CATEGORY"]].index)]
    return t.sort_values(["source_id", "condition", "variant"]).reset_index(drop=True)


class Block:
    """Grid controls and scrambles of one image set, with category-balanced bookkeeping."""

    def __init__(self, df, label):
        self.df = df
        self.objects = sorted(df.source_id.unique())
        oi = {o: i for i, o in enumerate(self.objects)}
        self.obj = df.source_id.map(oi).to_numpy()
        self.is_grid = (df.condition == "cutG_intact").to_numpy()
        self.y = df.category.map(C.CAT_TO_IDX).to_numpy()
        self.obj_cat = df.drop_duplicates("source_id").set_index("source_id").loc[self.objects, "category"].to_numpy()
        self.cats = sorted(set(self.obj_cat))
        self.grid_of_obj = np.array([np.where(self.is_grid & (self.obj == i))[0][0] for i in range(len(self.objects))])
        S = np.zeros((len(df), len(self.objects)), np.float32)
        for i in range(len(self.objects)):
            idx = np.where(~self.is_grid & (self.obj == i))[0]
            S[idx, i] = 1.0 / len(idx)
        self.SCR = S
        t0 = time.time()
        self.imgs = load_imgs(df.path.tolist())
        print(f"  {label}: {len(self.objects)} objects in {len(self.cats)} categories, {len(df)} images "
              f"({self.is_grid.sum()} grid) loaded in {time.time() - t0:0.0f} s")

    def weights(self, n_boot, seed_key):
        rng = np.random.default_rng(C.derive_seed(seed_key))
        n = len(self.objects)
        w0, W = np.zeros(n, np.float32), np.zeros((n_boot, n), np.float32)
        for c in self.cats:
            g = np.where(self.obj_cat == c)[0]
            s = 1.0 / (len(g) * len(self.cats))
            w0[g] = s
            draws = rng.integers(0, len(g), size=(n_boot, len(g)))
            cnt = np.zeros((n_boot, len(g)), np.float32)
            np.add.at(cnt, (np.repeat(np.arange(n_boot), len(g)), draws.ravel()), 1)
            W[:, g] = cnt * s
        return w0, W


fc = pd.read_csv(C.STIM_DRIVE / "frozen_calib_manifest.csv", low_memory=False)
LOC = Block(configural_frame(fc, "localizer"), "calibration localizer half")
assert not LOC.df.path.map(C.is_holdout_path).any()
hm = pd.read_csv(C.STIM_DRIVE / f"{C.HOLDOUT_STAGE}_manifest.csv", low_memory=False)
spec = C.load_json(C.HOLDOUT_SPEC)
assert C.fingerprint_files(C.STIM_LOCAL, hm.path.tolist()) == spec["fingerprint"], "Hold-out files changed since locking."
HOLD = Block(configural_frame(hm), "hold-out")
s1 = pd.read_csv(C.STIM_DRIVE / "stage1_manifest.csv", low_memory=False)
cue = s1[(s1.condition == "ref_cue-conflict") & (s1.category != s1.texture_category)].sort_values("path").reset_index(drop=True)
cue = cue[cue.category.isin(C.CATEGORIES) & cue.texture_category.isin(C.CATEGORIES)].reset_index(drop=True)
CUE_IMGS = load_imgs(cue.path.tolist())
CUE_SHAPE = cue.category.map(C.CAT_TO_IDX).to_numpy()
CUE_TEX = cue.texture_category.map(C.CAT_TO_IDX).to_numpy()
print(f"  cue-conflict: {len(cue)} images with different shape and texture categories")


# =============================================================================
# 2. Models and batched lesion evaluation
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
        if name == "resnet50_swsl":
            self.stages = [("stem", nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool)), ("layer1", m.layer1),
                           ("layer2", m.layer2), ("layer3", m.layer3), ("layer4", m.layer4)]
            self.head = lambda x: m.forward_head(x)
        else:
            self.stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.head = m.decoder
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


ROLE = {}            # offline test only: channel tuple -> role, filled when sets are known


def fake_decide(imgs, conds, idx):
    rng = np.random.default_rng(len(conds) * 7 + len(idx))
    out = np.zeros((len(conds), len(idx)), np.int8)
    for k, (ch, mode) in enumerate(conds):
        role = "intact" if len(ch) == 0 else ROLE.get(tuple(int(c) for c in ch), "random")
        if imgs is CUE_IMGS:
            ps, pt = dict(intact=(0.5, 0.3), configural=(0.3, 0.5), local=(0.6, 0.2)).get(role, (0.45, 0.33))
            u = rng.random(len(idx))
            other = (CUE_SHAPE[idx] + 1 + rng.integers(0, 14, len(idx))) % 16
            other = np.where(other == CUE_TEX[idx], (other + 1) % 16, other)
            out[k] = np.where(u < ps, CUE_SHAPE[idx], np.where(u < ps + pt, CUE_TEX[idx], other))
        else:
            blk = HOLD if imgs is HOLD.imgs else LOC
            pg, pr = dict(intact=(0.6, 0.3), configural=(0.5, 0.48), local=(0.5, 0.15)).get(role, (0.5, 0.3 + rng.normal(0, 0.02)))
            p = np.where(blk.is_grid[idx], pg, pr)
            yy = blk.y[idx]
            out[k] = np.where(rng.random(len(idx)) < p, yy, (yy + 1) % 16)
    return out


@torch.no_grad()
def decide(net, site, imgs, conds, mean_act, idx=None):
    """16-way decisions (n_conds, n_images) for a list of (channels, mode); an empty channel list is the
    intact network. Several conditions share one suffix call."""
    idx = np.arange(len(imgs)) if idx is None else idx
    if FAKE:
        return fake_decide(imgs, conds, idx)
    out = np.zeros((len(conds), len(idx)), np.int8)
    mv = torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
    B = CONFIG["BATCH"]
    for b0 in range(0, len(idx), B):
        sel = idx[b0:b0 + B]
        a = net.prefix(norm(imgs[sel]), site)
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
            logits = net.suffix(aa.flatten(0, 1), site)
            pred = (torch.softmax(logits.float(), 1) @ M16).argmax(1).view(len(chunk), nb)
            out[s0:s0 + len(chunk), b0:b0 + nb] = pred.cpu().numpy()
    return out


def grid_accuracy(block, pred_grid_rows):
    """Category-balanced grid-control accuracy from decisions on the block's grid images (object order)."""
    corr = (pred_grid_rows == block.y[block.grid_of_obj][None, :]).astype(np.float64)
    out = []
    for r in corr:
        out.append(pd.Series(r).groupby(block.obj_cat).mean().mean())
    return np.array(out)


def damages(net, site, sets, mean_act):
    """Drop in grid-control accuracy on the localizer half (zero ablation) for each set."""
    conds = [([], "zero")] + [(s, "zero") for s in sets]
    acc = grid_accuracy(LOC, decide(net, site, LOC.imgs, conds, mean_act, idx=LOC.grid_of_obj))
    return acc[0] - acc[1:]


# =============================================================================
# 3. Part A, selection on calibration data (written before the hold-out is used)
# =============================================================================
R, SITE = CONFIG["RESNET"], SETS["models"][CONFIG["RESNET"]]["primary_site"]
E_R = SETS["models"][R]["sites"][SITE]
MEAN_R = E_R["mean_activation"]
cs = SETS["models"][CONFIG["CORNET"]]["primary_site"]
TARGET = float(SETS["models"][CONFIG["CORNET"]]["sites"][cs]["damage_configural"])
SMALL_PATH = RES / f"sets_resnet_small{sfx}.json"
net_r = None

if SMALL_PATH.exists():
    SMALL = C.load_json(SMALL_PATH)
    print(f"\n[A] small ResNet-50 sets loaded ({SMALL_PATH.name})")
else:
    print(f"\n[A] selecting small ResNet-50 (SWSL) sets on the calibration localizer half | target damage {TARGET:0.4f}")
    net_r = Split(R)
    sc = SCORES[(SCORES.model == R) & (SCORES.site == SITE)].sort_values("channel")
    I_loc = sc.I_CR_localizer.to_numpy()
    local_score = sc.local_score.to_numpy()
    n_ch = len(sc)
    ranked = np.argsort(-np.nan_to_num(I_loc, nan=-np.inf))
    kmax = len(E_R["configural"])
    t0 = time.time()
    dmg_k = damages(net_r, SITE, [ranked[:k] for k in range(1, kmax + 1)], MEAN_R)
    k_star = int(np.argmin(np.abs(dmg_k - TARGET))) + 1
    conf = np.sort(ranked[:k_star])
    d_conf = float(dmg_k[k_star - 1])
    print(f"  damage scan k = 1 to {kmax} in {time.time() - t0:0.0f} s | k = {k_star}, damage {d_conf:0.4f}")
    elig = np.where(I_loc <= np.nanmedian(I_loc))[0]
    lranked = elig[np.argsort(-local_score[elig])]
    kl_max = min(len(lranked), max(4 * k_star, 50))
    dmg_l = damages(net_r, SITE, [lranked[:k] for k in range(1, kl_max + 1)], MEAN_R)
    kl = int(np.argmin(np.abs(dmg_l - d_conf))) + 1
    local = np.sort(lranked[:kl])
    print(f"  local set: {kl} channels, damage {dmg_l[kl - 1]:0.4f}")
    rng = np.random.default_rng(C.derive_seed(R, SITE, "small_size_matched"))
    size_sets = [np.sort(rng.choice(n_ch, k_star, replace=False)) for _ in range(CONFIG["N_SIZE"])]
    d_size = damages(net_r, SITE, size_sets, MEAN_R)
    rng = np.random.default_rng(C.derive_seed(R, SITE, "small_damage_pool"))
    tried, dts, within = [], [], []
    while len(tried) < CONFIG["DAMAGE_POOL"] and len(within) < CONFIG["N_DAMAGE"]:
        block = [np.sort(rng.choice(n_ch, k_star, replace=False)) for _ in range(100)]
        for s_, dd in zip(block, damages(net_r, SITE, block, MEAN_R)):
            tried.append(s_)
            dts.append(float(dd))
            if abs(dd - d_conf) <= CONFIG["DAMAGE_TOL"] and len(within) < CONFIG["N_DAMAGE"]:
                within.append(len(tried) - 1)
        print(f"\r  damage-matched search: {len(within)}/{CONFIG['N_DAMAGE']} within {CONFIG['DAMAGE_TOL']} "
              f"after {len(tried)} sets", end="")
    print()
    dts = np.array(dts)
    if len(within) >= CONFIG["N_DAMAGE"]:
        chosen, how = within[:CONFIG["N_DAMAGE"]], "within tolerance"
    else:
        chosen, how = list(np.argsort(np.abs(dts - d_conf))[:CONFIG["N_DAMAGE"]]), "closest (tolerance not reached)"
    SMALL = dict(site=SITE, target_damage=TARGET, k=k_star, damage_scan=dmg_k.tolist(),
                 configural=conf.tolist(), damage_configural=d_conf,
                 local=local.tolist(), damage_local=float(dmg_l[kl - 1]),
                 random_size_matched=[s_.tolist() for s_ in size_sets], damage_random_size_matched=d_size.tolist(),
                 random_damage_matched=[tried[i].tolist() for i in chosen], damage_random_damage_matched=dts[chosen].tolist(),
                 damage_matching=dict(rule=how, sets_tried=len(tried)),
                 created=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.save_json(SMALL, SMALL_PATH)
small_sha = hashlib.sha256(SMALL_PATH.read_bytes()).hexdigest()
if FAKE:
    for _n in CONFIG["MODELS"]:
        _e = SETS["models"][_n]["sites"][SETS["models"][_n]["primary_site"]]
        ROLE[tuple(_e["configural"])], ROLE[tuple(_e["local"])] = "configural", "local"
    ROLE[tuple(SMALL["configural"])], ROLE[tuple(SMALL["local"])] = "configural", "local"
print(f"  sets file SHA-256 {small_sha} | k = {SMALL['k']} ({100 * SMALL['k'] / len(MEAN_R):0.1f}% of {SITE}) | "
      f"damage configural {SMALL['damage_configural']:0.3f}, local {SMALL['damage_local']:0.3f} ({len(SMALL['local'])} channels) | "
      f"damage-matched: {SMALL['damage_matching']['rule']}")

# =============================================================================
# 4. Part A, hold-out test
# =============================================================================
LAB_A = ["intact", "configural", "local"] + [f"size {i + 1}" for i in range(len(SMALL["random_size_matched"]))] \
        + [f"damage {i + 1}" for i in range(len(SMALL["random_damage_matched"]))]
SETS_A = [[], SMALL["configural"], SMALL["local"]] + SMALL["random_size_matched"] + SMALL["random_damage_matched"]
DEC_A = {}
for mode in ("zero", "mean"):
    ck = CKPT / f"holdout_{mode}{sfx}.npz"
    if ck.exists():
        DEC_A[mode] = np.load(ck)["pred"]
        print(f"  hold-out, {mode} ablation: loaded from checkpoint")
        continue
    if net_r is None and not FAKE:
        net_r = Split(R)
    C.check_holdout_access(HOLD.df.path.tolist(), R)         # logs the run
    t0 = time.time()
    DEC_A[mode] = decide(net_r, SITE, HOLD.imgs, [(s_, mode) for s_ in SETS_A], MEAN_R)
    np.savez_compressed(ck, pred=DEC_A[mode], labels=np.array(LAB_A, dtype=object))
    print(f"  hold-out, {mode} ablation: {len(SETS_A)} conditions x {len(HOLD.df)} images in {(time.time() - t0) / 60:0.1f} min")

w0_h, W_h = HOLD.weights(CONFIG["N_BOOT"], "stage11_holdout")


def acc_block(pred):
    corr = (pred == HOLD.y[None, :]).astype(np.float32)
    G = corr[:, HOLD.grid_of_obj].T
    S = (corr @ HOLD.SCR).T
    return w0_h @ G, w0_h @ S, W_h @ G, W_h @ S


def cr_(ag, asc):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ag - asc) / (ag - C.CHANCE_16)


def summ(point, boot):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return dict(estimate=float(point), lo=np.nan, hi=np.nan, p_two_sided=np.nan)
    lo, hi = np.quantile(b, [0.025, 0.975])
    p = min(1.0, 2 * min((1 + (b <= 0).sum()) / (len(b) + 1), (1 + (b >= 0).sum()) / (len(b) + 1)))
    return dict(estimate=float(point), lo=float(lo), hi=float(hi), p_two_sided=float(p))


ROWS_A, LEVEL_A = [], []
for mode in ("zero", "mean"):
    ag, asc, AG, AS = acc_block(DEC_A[mode])
    ag_lo = np.quantile(AG, 0.025, axis=0)
    floor = ag_lo < CONFIG["FLOOR"]
    crp, CRB = cr_(ag, asc), cr_(AG, AS)
    loss, LOSS = ag - asc, AG - AS
    for i, l in enumerate(LAB_A[:3]):
        LEVEL_A.append(dict(ablation=mode, condition=l, A_grid=float(ag[i]), A_grid_lo=float(ag_lo[i]), A_scr=float(asc[i]),
                            CR=float(crp[i]) if not floor[i] else np.nan, floor=bool(floor[i]), loss=float(loss[i])))
    for kind in ("size", "damage"):
        ix = [i for i, l in enumerate(LAB_A) if l.startswith(kind)]
        LEVEL_A.append(dict(ablation=mode, condition=f"random {kind}-matched (mean)", A_grid=float(ag[ix].mean()),
                            A_grid_lo=np.nan, A_scr=float(asc[ix].mean()), CR=float(np.nanmean(np.where(floor[ix], np.nan, crp[ix]))),
                            floor=False, loss=float(loss[ix].mean()), n_at_floor=int(floor[ix].sum()), n_sets=len(ix)))
    d_conf_p, D_CONF = crp[1] - crp[0], CRB[:, 1] - CRB[:, 0]
    if floor[1] or floor[0]:
        ROWS_A.append(dict(prediction="R1/R2", ablation=mode, contrast="configural lesion at floor", estimate=np.nan,
                           lo=np.nan, hi=np.nan, held=False))
        continue
    for kind in ("size", "damage"):
        ix = [i for i, l in enumerate(LAB_A) if l.startswith(kind) and not floor[i]]
        if not ix:
            ROWS_A.append(dict(prediction="R1", ablation=mode, contrast=f"all {kind}-matched sets at floor", held=False))
            continue
        rp = np.mean(crp[ix] - crp[0])
        RB = np.mean(CRB[:, ix] - CRB[:, [0]], axis=1)
        s = summ(d_conf_p - rp, D_CONF - RB)
        row = dict(prediction="R1", ablation=mode, contrast=f"dCR configural minus mean dCR {kind}-matched random", **s,
                   n_random=len([l for l in LAB_A if l.startswith(kind)]), n_random_used=len(ix))
        if kind == "damage":
            p5 = float(np.quantile(crp[ix] - crp[0], 0.05))
            row.update(random_5th_percentile=p5, below_5th=bool(d_conf_p < p5))
        row["held"] = bool(s["hi"] < 0 and row.get("below_5th", True))
        ROWS_A.append(row)
        sl = summ(loss[1] - loss[0] - np.mean(loss[ix] - loss[0]),
                  (LOSS[:, 1] - LOSS[:, 0]) - np.mean(LOSS[:, ix] - LOSS[:, [0]], axis=1))
        ROWS_A.append(dict(prediction="R1 (absolute loss)", ablation=mode,
                           contrast=f"loss change configural minus {kind}-matched random", **sl, held=bool(sl["hi"] < 0)))
    if floor[2]:
        ROWS_A.append(dict(prediction="R2", ablation=mode, contrast="local lesion at floor", held=False))
    else:
        s = summ(crp[1] - crp[2], CRB[:, 1] - CRB[:, 2])
        ROWS_A.append(dict(prediction="R2", ablation=mode, contrast="dCR configural minus dCR local", **s, held=bool(s["hi"] < 0)))
    sl = summ(loss[1] - loss[2], LOSS[:, 1] - LOSS[:, 2])
    ROWS_A.append(dict(prediction="R2 (absolute loss)", ablation=mode, contrast="loss configural minus loss local", **sl,
                       held=bool(sl["hi"] < 0)))
del net_r
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# 5. Part B, shape bias on cue-conflict images
# =============================================================================
rng = np.random.default_rng(C.derive_seed("stage11_cue_bootstrap"))
n_cue = len(cue)
WB_c = np.zeros((CONFIG["N_BOOT"], n_cue), np.float32)
for c in sorted(set(CUE_SHAPE)):
    g = np.where(CUE_SHAPE == c)[0]
    draws = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
    cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
    np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), draws.ravel()), 1)
    WB_c[:, g] = cnt

ROWS_B, LEVEL_B = [], []
for name in CONFIG["MODELS"]:
    site = SETS["models"][name]["primary_site"]
    e = SETS["models"][name]["sites"][site]
    groups = [("planned", e["configural"], e["local"], e["random_damage_matched"][:CONFIG["N_RANDOM_CUE"]])]
    if name == R:
        groups.append(("small", SMALL["configural"], SMALL["local"], SMALL["random_damage_matched"][:CONFIG["N_RANDOM_CUE"]]))
    labels, conds = ["intact"], [([], "zero")]
    for gname, conf, loc, rnd in groups:
        for mode in ("zero", "mean"):
            labels += [f"{gname} configural {mode}", f"{gname} local {mode}"] + \
                      [f"{gname} random {i + 1} {mode}" for i in range(len(rnd))]
            conds += [(conf, mode), (loc, mode)] + [(r_, mode) for r_ in rnd]
    ck = CKPT / f"cue_{name}{sfx}.npz"
    if ck.exists():
        pred = np.load(ck)["pred"]
        print(f"\n[B] {C.MODEL_INFO[name][0]}: loaded from checkpoint")
    else:
        net = None if FAKE else Split(name)
        t0 = time.time()
        pred = decide(net, site, CUE_IMGS, conds, e["mean_activation"])
        np.savez_compressed(ck, pred=pred, labels=np.array(labels, dtype=object))
        print(f"\n[B] {C.MODEL_INFO[name][0]}: {len(conds)} conditions x {n_cue} cue-conflict images in "
              f"{(time.time() - t0) / 60:0.1f} min")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shape_hit = (pred == CUE_SHAPE[None, :]).astype(np.float32)
    tex_hit = (pred == CUE_TEX[None, :]).astype(np.float32)
    dec = shape_hit + tex_hit
    sb = shape_hit.sum(1) / np.maximum(dec.sum(1), 1)
    SB = (WB_c @ shape_hit.T) / np.maximum(WB_c @ dec.T, 1)
    frac = dec.mean(1)
    FRAC = (WB_c @ dec.T) / WB_c.sum(1, keepdims=True)
    estimable = np.quantile(FRAC, 0.025, axis=0) >= CONFIG["MIN_DECIDED"]
    ix = {l: i for i, l in enumerate(labels)}
    for l in labels:
        i = ix[l]
        lo, hi = np.quantile(SB[:, i], [0.025, 0.975])
        LEVEL_B.append(dict(model=name, condition=l, shape_bias=float(sb[i]), lo=float(lo), hi=float(hi),
                            decided_fraction=float(frac[i]), estimable=bool(estimable[i])))
    for gname, *_ in groups:
        for mode in ("zero", "mean"):
            ci_, li = ix[f"{gname} configural {mode}"], ix[f"{gname} local {mode}"]
            rix = [ix[l] for l in labels if l.startswith(f"{gname} random") and l.endswith(mode) and estimable[ix[l]]]
            base = SB[:, 0]
            if not (estimable[0] and rix):
                continue
            dR, DR = np.mean(sb[rix] - sb[0]), np.mean(SB[:, rix] - base[:, None], axis=1)
            for pred_id, cond_i, rule in [("S1 configural minus random", ci_, "upper<0"), ("S2 local minus random", li, "lower>0")]:
                if not estimable[cond_i]:
                    ROWS_B.append(dict(model=name, sets=gname, ablation=mode, prediction=pred_id, note="not estimable"))
                    continue
                s = summ(sb[cond_i] - sb[0] - dR, SB[:, cond_i] - base - DR)
                held = s["hi"] < 0 if rule == "upper<0" else s["lo"] > 0
                ROWS_B.append(dict(model=name, sets=gname, ablation=mode, prediction=pred_id, **s, held=bool(held),
                                   n_random_used=len(rix)))
            if estimable[ci_] and estimable[li]:
                s = summ(sb[li] - sb[ci_], SB[:, li] - SB[:, ci_])
                ROWS_B.append(dict(model=name, sets=gname, ablation=mode, prediction="S3 local minus configural", **s,
                                   held=bool(s["lo"] > 0)))
    print(f"  intact shape bias {sb[0]:0.3f} (decided {frac[0]:0.3f}) | planned configural zero "
          f"{sb[ix['planned configural zero']]:0.3f}, local zero {sb[ix['planned local zero']]:0.3f}")

# =============================================================================
# 6. Save and summarise
# =============================================================================
pd.DataFrame(ROWS_A).to_csv(RES / f"resnet_small_contrasts{sfx}.csv", index=False)
pd.DataFrame(LEVEL_A).to_csv(RES / f"resnet_small_levels{sfx}.csv", index=False)
pd.DataFrame(ROWS_B).to_csv(RES / f"shape_bias_contrasts{sfx}.csv", index=False)
pd.DataFrame(LEVEL_B).to_csv(RES / f"shape_bias_levels{sfx}.csv", index=False)
C.save_json(dict(config=CONFIG, small_sets_sha256=small_sha, k=SMALL["k"], target_damage=TARGET,
                 n_holdout_objects=len(HOLD.objects), n_cue_conflict=n_cue, lesion_sets_sha256=sets_sha,
                 run_at=time.strftime("%Y-%m-%d %H:%M:%S")), RES / f"stage11_summary{sfx}.json")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\nPart A, levels:")
print(pd.DataFrame(LEVEL_A).round(3).to_string(index=False))
print("\nPart A, predictions (exploratory):")
print(pd.DataFrame(ROWS_A).round(3).to_string(index=False))
print("\nPart B, predictions (exploratory):")
print(pd.DataFrame(ROWS_B).round(3).to_string(index=False))
if IN_COLAB:
    zp = Path("/content/stage11_results")
    shutil.make_archive(str(zp), "zip", root_dir=str(RES.parent), base_dir=RES.name)
    print(f"\nResults zipped to {zp}.zip (download it and send it over).")
print(f"\nSaved to {RES}")
