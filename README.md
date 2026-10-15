# Configural lesions in deep networks: preregistration

Frozen analysis plan for a computational study of integrative visual agnosia (the syndrome of
Dr P in Sacks, 1985) in deep networks. The hold-out set was generated and fingerprinted first,
then the analysis plan was frozen, before any study model was evaluated on the hold-out set.

| Item | Value |
| --- | --- |
| Hold-out created | 2026-10-15 11:56:31 (UTC) |
| Plan frozen at | 2026-10-15 13:07:24 (UTC) |
| Plan SHA-256 | `dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c` |
| Hold-out fingerprint | `91de2ec2b23333075e1b51e5157cd1975f1ba1ccff578cb13f56ba9cca8c86bf` |
| Release `preregistration-v1` | timestamped by GitHub on creation |

## Files

| File | Contents |
| --- | --- |
| `analysis_plan.md`, `analysis_plan.pdf` | The full plan: hypotheses, measures, unit selection, lesion protocol, statistics, failure criteria, design history |
| `analysis_plan_frozen.json` | The machine-readable plan the analysis code reads, with its SHA-256 and the hold-out fingerprint |
| `holdout_spec.json` | Hold-out generation record: frozen stimulus design, selection rules, categories, replication criteria |
| `holdout_manifest.csv` | Every hold-out image path, for recomputing the fingerprint |
| `holdout_access_log.jsonl` | Every model run on hold-out images, with time and model. Before the freeze, only the local-only BagNet controls appear |
| `stage1_gate_report.json`, `stage1b_report.json`, `holdout_replication.json` | Stimulus validation results on calibration and hold-out data |
| `stage2_selection.json`, `stage2_summary.csv` | Model selection results on the calibration localizer half |
| `agnosia_common.py` | Shared code, including the hold-out lock and the fingerprint function |

## Verifying the plan

```python
import json, hashlib
f = json.load(open("analysis_plan_frozen.json"))
body = json.dumps(f["plan"], sort_keys=True, default=str)
assert hashlib.sha256(body.encode()).hexdigest() == f["plan_sha256"]
```

The hold-out fingerprint is a SHA-256 over the sorted relative paths in `holdout_manifest.csv`
and the bytes of each image (`fingerprint_files` in `agnosia_common.py`). The images will be
released with the paper.
