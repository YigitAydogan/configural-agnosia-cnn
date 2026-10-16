# Deviations after the freeze: clarifications to the confirmatory analyses

Plan SHA-256: `dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c`. Amendment A1 SHA-256: `6dd4a7445862ba90f392a6028623abdf4d98340ada89fa8ff6f396bc5b814e9f`.

Each clarification below was written into the header of the analysis script before that script was first run on the hold-out set. None changes a hypothesis, measure, unit set or threshold.

| Analysis | Point | Clarification |
| --- | --- | --- |
| Hold-out lesion tests (H1 to H3) | Decision rule | A hypothesis is supported when its interval rule is met and its Holm-adjusted two-sided bootstrap p is below 0.05 (the plan lists both without stating how they combine). H2 has two contrasts (size- and damage-matched controls); its p-value is the larger of the two (intersection-union test). |
| Hold-out lesion tests (H1 to H3) | Random control lesions at the floor | A random control set whose own grid-control accuracy is at the floor has no CR; it is left out of the random mean and of the 5th-percentile test, and the number left out is reported. A sensitivity analysis keeps those sets (CR computed whenever grid-control accuracy exceeds chance). |
| Agnosia signature tests (H4 to H7) | Hue labels | The dominant hue of an object is taken from its grid-control image (6 bins of 60 degrees over pixels with saturation above 0.2); objects with fewer than 100 saturated pixels have no hue label and are left out of the hue probe. |
| Agnosia signature tests (H4 to H7) | Control tasks | Hold-out objects are new, so a probe trained on labels shuffled at object level is at chance on them by construction. Selectivity is therefore measured on the calibration set by holding out one scramble variant per object, for the two scrambled-image probes; grid-control and multi-label probes have one image per object, so no control task is defined for them. |
| Agnosia signature tests (H4 to H7) | Resampling units and H4 p-value | Bootstrap units: hold-out objects stratified by category (H4, H5), the 160 reference image pairs stratified by category (H6), hold-out figure pairs stratified by the first object's category (H7). Probes are fitted once per condition, so intervals cover test-set sampling. H4's p-value is computed against the 0.05 margin; with two probes, the larger p is used. |

## Exploratory analyses added after the freeze (reported in the Methods, no confirmatory claim)

- Unit-level piece interactions (exact Walsh decomposition over piece subsets) and the split of the scramble effect into position-coded parts and interactions, calibration test half.
- Concentration of network decisions and the readout gap (probe minus network accuracy), from the saved hold-out decisions.
- Progressive lesioning with brief retraining (named in the plan as exploratory), with a numerical safeguard added after one track diverged, and its five-seed replication with a log-space loss.
- Navon letter probes, Frankenstein effects after lesions, per-category results and the twelve-model profile (all named in the plan as exploratory).
