# Configural lesions in deep networks

Analysis plan, code and results for the article *Selective channel lesions remove reliance on part
arrangement in deep networks* (Yiğit Aydoğan, Aberystwyth University; submitted to Cognitive Computation).

The study asks whether deep networks rely on how the parts of an object are arranged, and whether
lesioning the channels that carry this reliance reproduces one dissociation of integrative visual
agnosia: part information preserved, arrangement lost. Release `manuscript-v1` is the version of this
repository that the article describes.

## Preregistration

The hold-out set was generated and fingerprinted first, then the analysis plan was frozen, before
any study model was evaluated on the hold-out set.

| Item | Value |
| --- | --- |
| Hold-out created | 2026-10-15 11:56:31 (UTC) |
| Plan frozen at | 2026-10-15 13:07:24 (UTC) |
| Plan SHA-256 | `dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c` |
| Hold-out fingerprint | `91de2ec2b23333075e1b51e5157cd1975f1ba1ccff578cb13f56ba9cca8c86bf` |
| Releases `preregistration-v1`, `-v1.1`, `-v1.2` | timestamped by GitHub on creation; unchanged since |

| File | Contents |
| --- | --- |
| `analysis_plan.md`, `analysis_plan.pdf` | The full plan: hypotheses, measures, unit selection, lesion protocol, statistics, failure criteria, design history |
| `analysis_plan_frozen.json` | The machine-readable plan the analysis code reads, with its SHA-256 and the hold-out fingerprint |
| `amendment_01.md`, `amendment_01.json` | The one amendment to the unit score, made on calibration data before testing |
| `holdout_spec.json` | Hold-out generation record: frozen stimulus design, selection rules, categories, replication criteria |
| `holdout_manifest.csv` | Every hold-out image path, for recomputing the fingerprint |
| `holdout_access_log.jsonl` | Every model run on hold-out images, with time and model |
| `lesion_sets_A1.json` | The lesion sets fixed before testing |
| `stage1_gate_report.json`, `stage1b_report.json`, `holdout_replication.json` | Stimulus validation results on calibration and hold-out data |
| `stage2_selection.json`, `stage2_summary.csv` | Model selection results on the calibration localizer half |
| `agnosia_common.py` | Shared code, including the hold-out lock and the fingerprint function |

### Verifying the plan

```python
import json, hashlib
f = json.load(open("analysis_plan_frozen.json"))
body = json.dumps(f["plan"], sort_keys=True, default=str)
assert hashlib.sha256(body.encode()).hexdigest() == f["plan_sha256"]
```

The hold-out fingerprint is a SHA-256 over the sorted relative paths in `holdout_manifest.csv` and the
bytes of each image (`fingerprint_files` in `agnosia_common.py`).

## Analysis code and results

| Folder | Contents |
| --- | --- |
| `code/` | Every analysis script in run order, each a self-contained Colab cell that embeds `agnosia_common.py`: `stage1_stimuli.py` to `stage9_final.py` (planned and exploratory analyses), `stage10_pieces_position.py`, `stage11_resnet_small_shape_bias.py` and `stage12_resnet50_supervised.py` (follow-up analyses, exploratory), the plan freeze, `figure1_stimuli_colab.py` and `render_paper_figs.py` |
| `results/` | All result tables (csv, json, tex) per analysis step. Large tables are gzipped (`.csv.gz`); `pandas.read_csv` reads them directly |
| `plan/deviations_after_freeze.md` | Every point fixed after the plan was frozen, and the exploratory analyses added |

The follow-up scripts were written, with their predictions in the script header, before each first ran.

### Where the article's figures and tables come from

| Article | Result files |
| --- | --- |
| Figure 1 | `code/figure1_stimuli_colab.py` |
| Figure 2, lesion accuracies | `results/stage9/figure1_accuracies.csv` |
| Figure 3, dose-response | `results/stage5/dose_response.csv` |
| Figure 4, readout collapse and retraining | `results/stage6b/readout_measures.csv`, `results/stage9/figure3_retraining_levels.csv` |
| Figure 5, piece interactions | `results/stage4a/`, `results/stage4b/` |
| Figure 6 and Table 6, follow-up lesions | `results/followup/followup_accuracies.csv`, `results/stage11/`, `results/stage12/` |
| Table 7, further follow-up predictions | `results/stage10/`, `results/stage11/` |

`python code/render_paper_figs.py` redraws Figures 2 to 6 from these tables into `figures_paper/`.

## Stimuli

The stimulus images are not stored here. They are regenerated from COCO 2017 and the
model-versus-human benchmark by `code/stage1_stimuli.py` and `code/stage1c_holdout.py`;
`holdout_manifest.csv` and the fingerprint function in `agnosia_common.py` check that a regenerated
hold-out set is identical to the one tested.
