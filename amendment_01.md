# Amendment A1: unit selection score

**Date:** 15 October 2026, after the plan freeze and before any study model touched the hold-out set.
**Amends plan:** SHA-256 `dc1a1b0e80b41dc3488c14bed7f96e4df281525eb112352a8533363360c48c6c`
**Machine-readable version:** `amendment_01.json`, SHA-256 `6dd4a7445862ba90f392a6028623abdf4d98340ada89fa8ff6f396bc5b814e9f`

## What changes

The primary score for selecting configural channels changes from an absolute difference to the channel's effect on configural reliance itself.

Original: I_conf(c) = [P_grid minus P_grid without c] minus [P_scr minus P_scr without c], tested by a one-sided t-test across objects.

Amended: with probability-based configural reliance CR_p = 1 minus (P_scr minus 1/16) / (P_grid minus 1/16), computed from mean true-category probabilities over localizer objects (scrambles averaged within each object first),

I_CR(c) = CR_p(intact) minus CR_p(channel c zero-ablated).

A positive I_CR means removing the channel lowers configural reliance. Significance uses a one-sided, category-stratified bootstrap over objects (10,000 resamples) with the same Benjamini-Hochberg control at q = 0.05 and the same 1 to 40% bounds. Every rule that ranked channels by I_conf now ranks by I_CR: the reliability check, the local-set eligibility (I_CR at or below the site median), the primary site and the dose-response sets.

Unchanged: the unit, zero ablation, the damage definition, the size- and damage-matched random controls (including the fallback to the 100 closest), the selective set, and every hypothesis, measure and test on the hold-out set.

## Why

Intact objects start from a higher probability than scrambled ones. A channel that lowers both by the same proportion therefore produces a larger absolute drop on intact objects and a positive I_conf, without any special role in configuration. The original score favoured generally important channels.

The calibration pilot showed this. In ResNet-50 (SWSL), layer3, the 60-channel configural set caused a damage of 0.246 that random sets of the same size rarely reached, and lesioning it raised configural reliance from 0.542 to 0.690, above random lesions (0.583, range 0.475 to 0.698). In CORnet-S, the configural set was 80 of 256 V4 channels and removed all recognition (grid-control accuracy 0.069).

The plan states that channels are selected by their causal effect on configural reliance. I_CR measures exactly that, in the same form as the primary outcome. The same baseline confound led to configural reliance replacing the absolute scramble loss at the model level (design history).

## Data seen before this amendment

Stage 3 calibration scores, sets and the exploratory pilot on the calibration test half. No study model has been evaluated on the hold-out set (`holdout_access_log.jsonl`).

## Commitments

1. This amendment is published before the new score is computed.
2. It is the only revision to unit selection. Whatever the new calibration pilot shows, the resulting sets go to the hold-out tests unchanged.
3. Results under the original score are reported in the supplement for comparison.
