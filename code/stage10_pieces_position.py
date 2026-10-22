"""
STAGE 10: Single pieces, global position and activation shift after lesions (exploratory)
CNN visual agnosia project

Single-cell script: paste into one Colab cell and run (GPU; about 1.5 to 2.5 hours on an A100, mostly
CORnet-S). Work is saved per model and per group of lesion conditions; if the runtime disconnects,
rerunning picks up where it stopped. At the end the results folder is zipped to /content for download.

Why. A pre-submission review raised four questions that the existing analyses cannot answer:
  (a) Is the "local" lesion local by an independent criterion, or only by how it was selected?
  (b) Does the lesioned network itself still recognise parts, beyond what a refitted probe can decode?
  (c) Is configural reliance tied to fixed canvas positions rather than to the arrangement of the object?
  (d) Does zero ablation push downstream activity outside the range the network was trained on?
Everything here is exploratory and carries no confirmatory claim. The predictions below were written
before this script was first run, and are reported whether they hold or fail.

Data (hold-out, the same 1421 objects as the hypothesis tests). Built in memory from the saved stimuli
and the saved object masks, so no new image is downloaded:
  pieces       each non-empty grid piece of the grid control shown alone in its own cell, everything
               else white (cells defined exactly as by the stimulus generator)
  shift        the grid region of the grid control and of one scramble per object (the qualifying
               variant with the largest layout change), shrunk to 70%, placed at the centre and at the
               four corners of a white canvas (4 px margin); relative arrangement kept, absolute
               position changed
  grid         the original grid controls, for the activation measures

Lesion conditions per model, at the primary site with the published sets (lesion_sets_A1.json):
intact; configural and local, each under zero and mean ablation; the first 20 damage-matched random
sets, each under zero and mean ablation (45 conditions). A lesion acts on the output of the site
(CORnet-S: the output of V4 after its internal recurrent steps), as in every earlier analysis.

Measures (category-balanced means; bootstrap over objects stratified by category, 10,000 resamples,
all conditions resampled jointly)
  piece accuracy   the network's own 16-way accuracy on single pieces, averaged within object first;
                   piece probability, the 16-way probability of the true category, as a graded companion
  shift            grid and scramble accuracy at the centre and at the corners (mean of four);
                   scramble loss = grid minus scramble accuracy (no floor rule needed); configural
                   reliance with the usual floor rule
  activation       penultimate features of the original grid controls, standardised by the intact
                   network's mean and standard deviation per feature: shift D = mean squared z-score
                   (1 for the intact network by construction); share of feature values outside the
                   intact network's 0.5th to 99.5th percentile; share of decisions on the most frequent
                   category (top share)

Predictions, written before the first run
  P1  Local lesion, single pieces: drop in piece accuracy after the local lesion minus the mean drop
      after the random lesions has upper 95% bound < 0 (the local lesion impairs parts beyond random
      damage of the same cost).
  P2  Configural lesion, single pieces: the same contrast for the configural lesion has lower bound
      > -0.05 (no impairment of parts beyond random damage, at the H4 margin).
  P3  Local versus configural, single pieces: drop after local minus drop after configural has upper
      bound < 0. Together with H3 (configural lowers configural reliance more than local), P3 would
      make the two lesions independently defined.
      P1 to P3 are judged within each model and ablation method; under zero ablation the output of
      ResNet-50 (SWSL) is expected to collapse (readout analysis), so for that model mean ablation is
      the informative test. A test is called uninformative if intact piece accuracy has lower bound
      below 2/16.
  P4  Position: in the intact networks, the scramble loss at the corners is at least half the scramble
      loss at the centre (lower bound of the ratio > 0.5): arrangement sensitivity survives a change
      of absolute position.
  P5  Position after lesion: the configural lesion lowers the scramble loss at the corners more than
      the random lesions (upper bound < 0), under the same ablation rule as P1 to P3.
  P6  Activation: for the configural and the local lesion, D after zero ablation exceeds D after
      mean ablation (lower bound of the difference > 0).
  P7  Activation and readout: across the 44 lesion conditions, D rises with the top share (Spearman
      correlation, lower bound > 0).

Outputs: results/stage10/*.csv|json, and /content/stage10_results.zip (results folder) for download.
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
import torch.nn.functional as F
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

CONFIG = dict(
    MODELS=[PLAN["models"]["primary"], PLAN["models"]["secondary"]],
    MIN_STRENGTH=PLAN["data"]["min_manip_strength"],
    EXCLUDED=PLAN["data"]["holdout_categories_excluded"],
    MIN_OBJECTS_PER_CATEGORY=20,
    GRID=3,
    MIN_PIECE_PX=40,
    N_RANDOM=20,
    SCALE=0.7,
    CORNER_MARGIN=4,
    FLOOR=2 * C.CHANCE_16,
    MARGIN=0.05,
    N_BOOT=10000,
    BATCH=128,
    COND_CHUNK=9,
)
if SMOKE:
    CONFIG.update(N_BOOT=300, BATCH=32, MIN_OBJECTS_PER_CATEGORY=2, N_RANDOM=3)

C.ensure_dirs()
C.set_seed()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True     # same numerical settings as the earlier lesion analyses
torch.backends.cudnn.allow_tf32 = True
DEVICE = C.get_device()
RES = C.RESULTS_DIR / "stage10"
CKPT = RES / "checkpoints"
for dd in [RES, CKPT]:
    dd.mkdir(parents=True, exist_ok=True)
print(f"Stage 10 (exploratory) | plan {FROZEN['plan_sha256'][:12]} | lesion sets {sets_sha[:12]} | device {DEVICE}")

# =============================================================================
# 1. Hold-out objects, masks and grid geometry
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
d = d[~d.category.isin(small)]

mz = np.load(C.RESULTS_DIR / "stage1c" / "holdout_masks.npz")
MASKS = dict(zip(mz["source_ids"].tolist(), mz["masks"]))
H = W = C.IMG_SIZE


def grid_geometry(mask):
    """Same geometry as the stimulus generator: (R0, C0, ch, cw)."""
    G = CONFIG["GRID"]
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    h, w = r1 - r0, c1 - c0
    ch, cw = min(int(np.ceil(h / G)), H // G), min(int(np.ceil(w / G)), W // G)
    R0 = int(np.clip(r0 - (G * ch - h) // 2, 0, H - G * ch))
    C0 = int(np.clip(c0 - (G * cw - w) // 2, 0, W - G * cw))
    return R0, C0, ch, cw


OBJ = []
for sid, g in d.groupby("source_id", sort=True):
    if sid not in MASKS:
        continue
    mask = MASKS[sid]
    R0, C0, ch, cw = grid_geometry(mask)
    cells = []
    for i in range(CONFIG["GRID"]):
        for j in range(CONFIG["GRID"]):
            y0, x0 = R0 + i * ch, C0 + j * cw
            if mask[y0:y0 + ch, x0:x0 + cw].sum() >= CONFIG["MIN_PIECE_PX"]:
                cells.append((y0, y0 + ch, x0, x0 + cw))
    scr = g[g.condition == "scrambled"].sort_values(["manip_strength", "variant"], ascending=[False, True])
    OBJ.append(dict(source_id=sid, category=g.category.iloc[0], grid=g[g.condition == "cutG_intact"].path.iloc[0],
                    scr=scr.path.iloc[0], region=(R0, R0 + CONFIG["GRID"] * ch, C0, C0 + CONFIG["GRID"] * cw),
                    cells=cells))
n_obj = len(OBJ)
assert n_obj > 0, "No hold-out objects with masks."
obj_cat = np.array([o["category"] for o in OBJ])
CATS = sorted(set(obj_cat))
y_obj = np.array([C.CAT_TO_IDX[c] for c in obj_cat])
missing = len(set(keep) - set(MASKS))
print(f"  {n_obj} objects in {len(CATS)} categories | {sum(len(o['cells']) for o in OBJ)} single pieces | "
      f"objects without a saved mask: {missing}")
SOURCE_PATHS = [o["grid"] for o in OBJ] + [o["scr"] for o in OBJ]

size = 64 if SMOKE else H
sc = size / H
t0 = time.time()
BASE = torch.from_numpy(np.stack([np.stack([np.array(Image.open(C.STIM_LOCAL / o[k]).convert("RGB").resize((size, size)),
                                                     dtype=np.uint8) for k in ("grid", "scr")]) for o in OBJ]))
BASE = BASE.permute(0, 1, 4, 2, 3).contiguous()          # (n_obj, 2, 3, size, size) uint8, CPU
print(f"  base images loaded in {time.time() - t0:0.0f} s")


def scaled(box):
    return tuple(int(round(v * sc)) for v in box)


POSITIONS = ["centre", "top left", "top right", "bottom left", "bottom right"]
SPECS = []                     # (kind, object index, parameter)
for i, o in enumerate(OBJ):
    for k in range(len(o["cells"])):
        SPECS.append(("piece", i, k))
    for arr in (0, 1):
        for p in range(len(POSITIONS)):
            SPECS.append(("shift", i, (arr, p)))
    SPECS.append(("grid", i, None))
SPEC_KIND = np.array([s[0] for s in SPECS])
SPEC_OBJ = np.array([s[1] for s in SPECS])
print(f"  {len(SPECS)} images per lesion condition")

MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
M16 = torch.zeros(1000, C.N_CAT, device=DEVICE)
for j, c in enumerate(C.CATEGORIES):
    M16[C.IMAGENET_16[c], j] = 1.0 / len(C.IMAGENET_16[c])


def build_batch(specs):
    """Construct a batch of images on the GPU; returns normalised floats (B, 3, size, size)."""
    out = torch.full((len(specs), 3, size, size), 255.0, device=DEVICE)
    for b, (kind, i, par) in enumerate(specs):
        o = OBJ[i]
        if kind == "grid":
            out[b] = BASE[i, 0].to(DEVICE).float()
        elif kind == "piece":
            y0, y1, x0, x1 = scaled(o["cells"][par])
            out[b, :, y0:y1, x0:x1] = BASE[i, 0, :, y0:y1, x0:x1].to(DEVICE).float()
        else:
            arr, p = par
            r0, r1, c0, c1 = scaled(o["region"])
            r0, c0 = max(r0, 0), max(c0, 0)
            crop = BASE[i, arr, :, r0:r1, c0:c1].to(DEVICE).float()[None]
            nh, nw = max(1, int(round(crop.shape[2] * CONFIG["SCALE"]))), max(1, int(round(crop.shape[3] * CONFIG["SCALE"])))
            small_img = F.interpolate(crop, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)[0]
            m = int(round(CONFIG["CORNER_MARGIN"] * sc))
            ys = {"centre": (size - nh) // 2, "top left": m, "top right": m,
                  "bottom left": size - nh - m, "bottom right": size - nh - m}[POSITIONS[p]]
            xs = {"centre": (size - nw) // 2, "top left": m, "top right": size - nw - m,
                  "bottom left": m, "bottom right": size - nw - m}[POSITIONS[p]]
            out[b, :, ys:ys + nh, xs:xs + nw] = small_img.clamp(0, 255)
    return (out / 255.0 - MEAN) / STD


def example_sheet(n_examples=3):
    """Every image built for the first few objects, saved as one PNG for visual checking."""
    rows = []
    for i in range(min(n_examples, n_obj)):
        sel = [s for s in SPECS if s[1] == i]
        x = build_batch(sel) * STD + MEAN
        rows.append((x.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy())
    w = max(len(r) for r in rows)
    sheet = np.full((len(rows) * (size + 4), w * (size + 4), 3), 128, np.uint8)
    for a, r in enumerate(rows):
        for b, im in enumerate(r):
            sheet[a * (size + 4):a * (size + 4) + size, b * (size + 4):b * (size + 4) + size] = im
    Image.fromarray(sheet).save(RES / f"example_images{sfx}.png")


example_sheet()


# =============================================================================
# 2. Models: prefix to the site, suffix to penultimate features and 16-way output
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
            self.feat = lambda x: m.forward_head(x, pre_logits=True)
            self.fc = m.fc
        else:
            self.stages = [("V1", m.V1), ("V2", m.V2), ("V4", m.V4), ("IT", m.IT)]
            self.feat = lambda x: m.decoder.flatten(m.decoder.avgpool(x))
            self.fc = m.decoder.linear
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
        p16 = torch.softmax(self.fc(f).float(), 1) @ M16
        p16 = p16 / p16.sum(1, keepdim=True)
        return f, p16


def conditions_for(name):
    mo = SETS["models"][name]
    e = mo["sites"][mo["primary_site"]]
    conds = [("intact", [], "zero")]
    for mode in ("zero", "mean"):
        conds += [(f"configural {mode}", e["configural"], mode), (f"local {mode}", e["local"], mode)]
        conds += [(f"random {k + 1} {mode}", e["random_damage_matched"][k], mode) for k in range(CONFIG["N_RANDOM"])]
    return mo["primary_site"], e["mean_activation"], conds


@torch.no_grad()
def intact_reference(net, site):
    """Per-feature mean, standard deviation and 0.5/99.5 percentiles of intact penultimate features on grid controls."""
    idx = np.where(SPEC_KIND == "grid")[0]
    feats = []
    for b0 in range(0, len(idx), CONFIG["BATCH"]):
        x = build_batch([SPECS[k] for k in idx[b0:b0 + CONFIG["BATCH"]]])
        f, _ = net.suffix(net.prefix(x, site), site)
        feats.append(f.cpu())
    Fm = torch.cat(feats)
    q = torch.quantile(Fm, torch.tensor([0.005, 0.995]), dim=0)
    return Fm.mean(0).to(DEVICE), Fm.std(0).clamp_min(1e-6).to(DEVICE), q[0].to(DEVICE), q[1].to(DEVICE)


@torch.no_grad()
def run_chunk(net, site, mean_act, conds, ref):
    """For each condition: 16-way decision and true-category probability for every image; activation shift D
    and outside share for grid-control images."""
    mu, sd, qlo, qhi = ref
    mv = torch.as_tensor(np.asarray(mean_act), device=DEVICE, dtype=torch.float32)
    n = len(SPECS)
    pred = np.zeros((len(conds), n), np.int8)
    ptrue = np.zeros((len(conds), n), np.float16)
    dist = np.full((len(conds), n), np.nan, np.float32)
    outs = np.full((len(conds), n), np.nan, np.float32)
    yb_all = torch.as_tensor(y_obj[SPEC_OBJ], device=DEVICE)
    for b0 in range(0, n, CONFIG["BATCH"]):
        sl = slice(b0, min(n, b0 + CONFIG["BATCH"]))
        a = net.prefix(build_batch(SPECS[sl]), site)
        yb = yb_all[sl]
        is_g = torch.as_tensor(SPEC_KIND[sl] == "grid", device=DEVICE)
        for k, (_, ch, mode) in enumerate(conds):
            aa = a
            if len(ch):
                aa = a.clone()
                chi = torch.as_tensor(np.asarray(ch, dtype=np.int64), device=DEVICE)
                if mode == "zero":
                    aa[:, chi] = 0
                else:
                    aa[:, chi] = mv[chi].to(aa.dtype).view(1, -1, 1, 1)
            f, p16 = net.suffix(aa, site)
            pred[k, sl] = p16.argmax(1).cpu().numpy()
            ptrue[k, sl] = p16.gather(1, yb[:, None])[:, 0].cpu().numpy()
            if is_g.any():
                z = (f - mu) / sd
                dd = (z ** 2).mean(1)
                oo = ((f < qlo) | (f > qhi)).float().mean(1)
                gi = np.where(SPEC_KIND[sl] == "grid")[0]
                dist[k, b0 + gi] = dd[is_g].cpu().numpy()
                outs[k, b0 + gi] = oo[is_g].cpu().numpy()
    return pred, ptrue, dist, outs


# =============================================================================
# 3. Run (checkpointed per model and group of conditions)
# =============================================================================
RUN = {}
for name in CONFIG["MODELS"]:
    label = C.MODEL_INFO[name][0]
    site, mean_act, conds = conditions_for(name)
    print(f"\n[{label}] site {site} | {len(conds)} lesion conditions")
    net, ref = None, None
    parts = []
    for c0 in range(0, len(conds), CONFIG["COND_CHUNK"]):
        chunk = conds[c0:c0 + CONFIG["COND_CHUNK"]]
        ck = CKPT / f"{name}_{c0:03d}{sfx}.npz"
        if ck.exists():
            z = np.load(ck, allow_pickle=True)
            parts.append((list(z["labels"]), z["pred"], z["ptrue"], z["dist"], z["outs"]))
            print(f"  conditions {c0 + 1} to {c0 + len(chunk)}: loaded from checkpoint")
            continue
        if net is None:
            C.check_holdout_access(SOURCE_PATHS, name)         # logs the run
            net = Split(name)
            ref = intact_reference(net, site)
        t0 = time.time()
        pr, pt, di, ou = run_chunk(net, site, mean_act, chunk, ref)
        labels = [c[0] for c in chunk]
        np.savez_compressed(ck, labels=np.array(labels, dtype=object), pred=pr, ptrue=pt, dist=di, outs=ou)
        parts.append((labels, pr, pt, di, ou))
        print(f"  conditions {c0 + 1} to {c0 + len(chunk)} in {(time.time() - t0) / 60:0.1f} min")
    RUN[name] = dict(labels=sum([p[0] for p in parts], []), pred=np.concatenate([p[1] for p in parts]),
                     ptrue=np.concatenate([p[2] for p in parts]).astype(np.float32),
                     dist=np.concatenate([p[3] for p in parts]), outs=np.concatenate([p[4] for p in parts]))
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# 4. Per-object values and bootstrap weights
# =============================================================================
rng = np.random.default_rng(C.derive_seed("stage10_bootstrap"))
W0 = np.zeros(n_obj, np.float32)
WB = np.zeros((CONFIG["N_BOOT"], n_obj), np.float32)
for c in CATS:
    g = np.where(obj_cat == c)[0]
    s = 1.0 / (len(g) * len(CATS))
    W0[g] = s
    draws = rng.integers(0, len(g), size=(CONFIG["N_BOOT"], len(g)))
    cnt = np.zeros((CONFIG["N_BOOT"], len(g)), np.float32)
    np.add.at(cnt, (np.repeat(np.arange(CONFIG["N_BOOT"]), len(g)), draws.ravel()), 1)
    WB[:, g] = cnt * s


def per_object(values, mask):
    """Mean of image values over the images of each object selected by mask -> (n_cond, n_obj)."""
    idx = np.where(mask)[0]
    M = np.zeros((len(idx), n_obj), np.float32)
    M[np.arange(len(idx)), SPEC_OBJ[idx]] = 1.0
    M /= np.maximum(M.sum(0, keepdims=True), 1)
    return values[:, idx].astype(np.float32) @ M


def agg(v):
    """(n_cond, n_obj) -> point (n_cond,), bootstrap (N_BOOT, n_cond)."""
    return v @ W0, WB @ v.T


def ci(boot):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    return (np.quantile(b, 0.025), np.quantile(b, 0.975)) if len(b) > 10 else (np.nan, np.nan)


ROWS_LEVEL, ROWS_CONTRAST, ROWS_DIAG = [], [], []


def contrast_row(model, test, ablation, point, boot, rule, informative=True, note=""):
    lo, hi = ci(boot)
    held = {"upper<0": hi < 0, "lower>-margin": lo > -CONFIG["MARGIN"], "lower>0.5": lo > 0.5, "lower>0": lo > 0}[rule]
    ROWS_CONTRAST.append(dict(model=model, prediction=test, ablation=ablation, estimate=float(point), lo=float(lo),
                              hi=float(hi), rule=rule, informative=bool(informative),
                              held=bool(held) if informative else None, note=note))


cr = lambda ag, asc: (ag - asc) / (ag - C.CHANCE_16)

for name in CONFIG["MODELS"]:
    label = C.MODEL_INFO[name][0]
    R = RUN[name]
    L = R["labels"]
    ix = {l: i for i, l in enumerate(L)}
    correct = (R["pred"] == y_obj[SPEC_OBJ][None, :]).astype(np.float32)

    # ---- single pieces
    pa, PA = agg(per_object(correct, SPEC_KIND == "piece"))
    pp, PP = agg(per_object(R["ptrue"], SPEC_KIND == "piece"))
    for l in L:
        i = ix[l]
        ROWS_LEVEL.append(dict(model=name, analysis="single pieces", condition=l, measure="piece accuracy",
                               value=float(pa[i]), lo=ci(PA[:, i])[0], hi=ci(PA[:, i])[1]))
        ROWS_LEVEL.append(dict(model=name, analysis="single pieces", condition=l, measure="piece probability",
                               value=float(pp[i]), lo=ci(PP[:, i])[0], hi=ci(PP[:, i])[1]))
    informative = ci(PA[:, ix["intact"]])[0] >= CONFIG["FLOOR"]
    for mode in ("zero", "mean"):
        rnd = [ix[f"random {k + 1} {mode}"] for k in range(CONFIG["N_RANDOM"])]
        for meas, (v, V) in [("piece accuracy", (pa, PA)), ("piece probability", (pp, PP))]:
            dl = lambda j: (v[j] - v[ix["intact"]], V[:, j] - V[:, ix["intact"]])
            dr = (np.mean([dl(j)[0] for j in rnd]), np.mean([dl(j)[1] for j in rnd], axis=0))
            dloc, dcon = dl(ix[f"local {mode}"]), dl(ix[f"configural {mode}"])
            note = meas + ("" if meas == "piece accuracy" else " (companion measure)")
            inf = informative or meas == "piece probability"
            contrast_row(name, "P1 local minus random", mode, dloc[0] - dr[0], dloc[1] - dr[1], "upper<0", inf, note)
            contrast_row(name, "P2 configural minus random", mode, dcon[0] - dr[0], dcon[1] - dr[1], "lower>-margin", inf, note)
            contrast_row(name, "P3 local minus configural", mode, dloc[0] - dcon[0], dloc[1] - dcon[1], "upper<0", inf, note)

    # ---- shift
    sh = np.where(SPEC_KIND == "shift")[0]
    arr = np.array([SPECS[k][2][0] for k in sh])
    pos = np.array([SPECS[k][2][1] for k in sh])
    vals = {}
    for a_, an in ((0, "grid"), (1, "scr")):
        for pname, psel in (("centre", pos == 0), ("corners", pos > 0)):
            m = np.zeros(len(SPECS), bool)
            m[sh[(arr == a_) & psel]] = True
            vals[(an, pname)] = agg(per_object(correct, m))
    for pname in ("centre", "corners"):
        ag, AG = vals[("grid", pname)]
        asr, AS = vals[("scr", pname)]
        for l in L:
            i = ix[l]
            agl = ci(AG[:, i])[0]
            for meas, v, V in [("grid accuracy", ag[i], AG[:, i]), ("scramble accuracy", asr[i], AS[:, i]),
                               ("scramble loss", ag[i] - asr[i], AG[:, i] - AS[:, i])]:
                ROWS_LEVEL.append(dict(model=name, analysis=f"shift, {pname}", condition=l, measure=meas, value=float(v),
                                       lo=ci(V)[0], hi=ci(V)[1]))
            floor = agl < CONFIG["FLOOR"]
            crv = np.nan if floor else cr(ag[i], asr[i])
            crb = cr(AG[:, i], AS[:, i]) if not floor else np.full(CONFIG["N_BOOT"], np.nan)
            ROWS_LEVEL.append(dict(model=name, analysis=f"shift, {pname}", condition=l, measure="configural reliance",
                                   value=float(crv), lo=ci(crb)[0], hi=ci(crb)[1], floor=bool(floor)))
    agc, AGc = vals[("grid", "centre")]
    asc, ASc = vals[("scr", "centre")]
    agk, AGk = vals[("grid", "corners")]
    ask, ASk = vals[("scr", "corners")]
    i0 = ix["intact"]
    lc, LC = agc[i0] - asc[i0], AGc[:, i0] - ASc[:, i0]
    lk, LK = agk[i0] - ask[i0], AGk[:, i0] - ASk[:, i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        contrast_row(name, "P4 corner loss over centre loss", "none", lk / lc, LK / LC, "lower>0.5", bool(ci(LC)[0] > 0),
                     "intact network; uninformative if the centre loss itself includes 0")
    for mode in ("zero", "mean"):
        rnd = [ix[f"random {k + 1} {mode}"] for k in range(CONFIG["N_RANDOM"])]
        loss_k = agk - ask
        LOSS_K = AGk - ASk
        dcon = loss_k[ix[f"configural {mode}"]] - loss_k[i0], LOSS_K[:, ix[f"configural {mode}"]] - LOSS_K[:, i0]
        dr = (np.mean([loss_k[j] - loss_k[i0] for j in rnd]), np.mean([LOSS_K[:, j] - LOSS_K[:, i0] for j in rnd], axis=0))
        contrast_row(name, "P5 configural minus random, corner scramble loss", mode, dcon[0] - dr[0], dcon[1] - dr[1],
                     "upper<0", True, "absolute scramble loss, no floor rule")

    # ---- activation shift and readout
    gm = SPEC_KIND == "grid"
    dD, DD = agg(per_object(np.nan_to_num(R["dist"]), gm))
    dO, DO = agg(per_object(np.nan_to_num(R["outs"]), gm))
    gidx = np.where(gm)[0]
    onehot = np.zeros((len(L), n_obj, C.N_CAT), np.float32)
    for k in range(len(L)):
        onehot[k, SPEC_OBJ[gidx], R["pred"][k, gidx]] = 1.0
    cnt0 = np.einsum("o,koc->kc", W0, onehot)
    top = cnt0.max(1) / cnt0.sum(1)
    cntB = np.einsum("bo,koc->bkc", WB, onehot)
    TOP = cntB.max(2) / cntB.sum(2)
    acc_g, ACC_G = agg(per_object(correct, gm))
    for l in L:
        i = ix[l]
        for meas, v, V in [("activation shift D", dD[i], DD[:, i]), ("share outside intact range", dO[i], DO[:, i]),
                           ("top share", top[i], TOP[:, i]), ("grid accuracy", acc_g[i], ACC_G[:, i])]:
            ROWS_DIAG.append(dict(model=name, condition=l, measure=meas, value=float(v), lo=ci(V)[0], hi=ci(V)[1]))
    for les in ("configural", "local"):
        a, b = ix[f"{les} zero"], ix[f"{les} mean"]
        contrast_row(name, f"P6 {les}: D zero minus D mean", "zero vs mean", dD[a] - dD[b], DD[:, a] - DD[:, b], "lower>0", True)
    les_ix = [i for i, l in enumerate(L) if l != "intact"]

    def spearman(x, y):
        rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        return np.corrcoef(rx, ry)[0, 1]

    rho = spearman(dD[les_ix], top[les_ix])
    RHO = np.array([spearman(DD[b, les_ix], TOP[b, les_ix]) for b in range(CONFIG["N_BOOT"])])
    contrast_row(name, "P7 Spearman of D with top share", "both", rho, RHO, "lower>0", True,
                 f"{len(les_ix)} lesion conditions")
    print(f"  {label}: intact piece accuracy {pa[i0]:0.3f} | centre and corner scramble loss {lc:0.3f}, {lk:0.3f} | "
          f"D configural zero {dD[ix['configural zero']]:0.2f}, mean {dD[ix['configural mean']]:0.2f}")

# =============================================================================
# 5. Save and summarise
# =============================================================================
levels = pd.DataFrame(ROWS_LEVEL)
contrasts = pd.DataFrame(ROWS_CONTRAST)
diag = pd.DataFrame(ROWS_DIAG)
levels.to_csv(RES / f"levels{sfx}.csv", index=False)
contrasts.to_csv(RES / f"predictions{sfx}.csv", index=False)
diag.to_csv(RES / f"activation_readout{sfx}.csv", index=False)
C.save_json(dict(config=CONFIG, n_objects=n_obj, n_images_per_condition=len(SPECS), lesion_sets_sha256=sets_sha,
                 positions=POSITIONS, run_at=time.strftime("%Y-%m-%d %H:%M:%S")), RES / f"stage10_summary{sfx}.json")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
show = contrasts[contrasts.note.str.contains("companion") == False] if len(contrasts) else contrasts
print("\nPredictions (exploratory):")
print(show[["model", "prediction", "ablation", "estimate", "lo", "hi", "informative", "held"]].round(3).to_string(index=False))

if IN_COLAB:
    zp = Path("/content/stage10_results")
    shutil.make_archive(str(zp), "zip", root_dir=str(RES.parent), base_dir=RES.name)
    print(f"\nResults zipped to {zp}.zip (download it and send it over); checkpoints are included.")
print(f"\nSaved to {RES}")
