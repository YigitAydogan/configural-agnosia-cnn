# Configural lesions in deep networks: analysis plan

Oct 15, 2026 · @Yiğit Aydoğan

## Purpose and status

This plan fixes every analysis that will be run on the hold-out set before any unit is selected or lesioned. It is frozen on Drive together with the hold-out fingerprint and timestamped in a GitHub repository; study models cannot be evaluated on the hold-out set until that freeze exists.

What is already done and fixed:

- **Stimuli.** The battery is frozen: Frankenstein with a 6 px horizontal cut, grid scramble with 12 px gaps, textured and silhouette renderings of every object.
- **Hold-out set.** 1500 COCO objects from images disjoint from the calibration set, fingerprint `91de2ec2b23333075e1b51e5157cd1975f1ba1ccff578cb13f56ba9cca8c86bf`, generated 15 October 2026. The local-only control (BagNet-33) replicated stimulus equivalence on it.
- **Models.** ResNet-50 (SWSL) is the primary model and CORnet-S the secondary, selected by a pre-specified rule on the localizer half of the calibration set.

The project has a single author, so every design decision below cites the literature it rests on rather than an individual's judgement. Analyses on the calibration set (Stages 3 to 4) are exploratory and may be iterated. Analyses on the hold-out set follow this document exactly. Any departure is recorded in the deviations log with its date and reason, and reported in the paper.

## Background and hypotheses

The central prediction is that removing a small set of configuration-dependent units makes a network behave like an integrative agnosic: intact objects become as hard as scrambled ones, while local evidence survives.

In Sacks's account (1985), Dr P described a glove as a continuous surface folded in on itself and a rose as a red convoluted form with a green attachment. He reported parts and surface features accurately but could not name the whole. Integrative agnosia has the same profile: parts are perceived but not combined into wholes ([Riddoch & Humphreys, 1987](https://en.wikipedia.org/wiki/Integrative_agnosia)). Patients represent a few parts independently but fail when categorisation depends on part relations, and are near chance with overlapping figures ([Behrmann & Williams, 2007](https://www.cmu.edu/dietrich/behrmannlab/Publications/Impairments%20in%20part%20whole%20representations%20of%20objects%20in%20two%20cases%20of%20integrative%20visual%20agnosia.pdf)). The integrative agnosic HJA lost the normal advantage of line drawings over silhouettes, consistent with internal contours fragmenting rather than supporting the whole ([Lawson & Humphreys, 1999](https://www.liverpool.ac.uk/~rlawson/HTML_TXT_Files/L&H-VisCog-1999.html)).

Standard CNNs are largely insensitive to object configuration ([Baker & Elder, 2022](https://doaj.org/article/33ba76d856ce4f14ac57cd060a6aadb8)), and our calibration data agree for Frankenstein stimuli. Grid scrambling, by contrast, costs the selected models far more than a local-only control. This gives a measurable configural component to remove.

The key outcome is configural reliance (CR): the scramble loss as a share of above-chance accuracy, defined in Outcome measures. A configural lesion should lower CR toward the local-only level. A random lesion of the same size should not.

| ID | Hypothesis | Prediction | Family |
| --- | --- | --- | --- |
| H1 | Both selected models rely on configuration on the hold-out set | CR excess over BagNet-33 above 0 | Primary |
| H2 | Configural units carry that reliance | CR falls more after the configural lesion than after size-matched and damage-matched random lesions | Primary |
| H3 | The deficit dissociates from local processing | CR falls more after the configural lesion than after a lesion of local units of equal damage | Primary |
| H4 | Local evidence survives, as in Dr P | Hue decoding and category decoding from scrambled images drop no more than after random lesions | Secondary |
| H5 | Errors become local | Error confusions move toward BagNet-33's confusion pattern more than after random lesions | Secondary |
| H6 | The line-drawing advantage shrinks, as in HJA | Edge-set minus silhouette-set accuracy falls more than after random lesions | Secondary |
| H7 | Overlapping figures suffer most | Overlap cost of a multi-object read-out rises more than after random lesions | Secondary |

Frankenstein effects are near zero in every model, so they carry no lesion hypothesis. They are reported descriptively as a model-human difference.

## Data, stimuli and models

Units are selected on the calibration set and every confirmatory test runs on the hold-out set, which shares no photograph with it.

| Set | Objects | Images | Use |
| --- | --- | --- | --- |
| Calibration, localizer half | 880 (800 textured) | about 13,300 | Model choice, unit scoring and selection |
| Calibration, test half | 879 (799 textured) | about 13,300 | Reliability of unit scores, lesion piloting |
| Hold-out | 1500 textured, 15 categories | 26,796 | All confirmatory tests (H1 to H7) |
| Public reference sets | 160 silhouettes, 160 edge drawings, 800 sketches, 1280 cue-conflict | 2400 | H6 and descriptive profile |

**Battery.** Each object appears intact, with a 6 px horizontal cut (control), as two Frankenstein variants, with a 3 by 3 grid of 12 px gaps (control), and in three grid scrambles. Textured objects keep their photo pixels; silhouettes are binary. Hold-out Navon and overlapping-figure images use new seeds and hold-out objects.

**Categories.** The 16 model-vs-human categories are read out by averaging softmax over their ImageNet classes. The hold-out set has no knives, so hold-out means cover 15 categories, weighted equally.

**Models.** ResNet-50 (SWSL) is primary and CORnet-S secondary. They were selected from nine candidates by the pre-specified Stage 2 rule (scramble excess over BagNet-33: 0.190 \[0.157, 0.224\] and 0.103 \[0.076, 0.130\]). BagNet-33 is the local-only reference. The other seven candidates are reported descriptively and do not enter the lesion analyses.

**Sites.** Lesions target channels at four stage outputs per model: layer1 to layer4 in ResNet-50 and V1, V2, V4 and IT in CORnet-S, whose names map onto the ventral stream.

## Outcome measures

The primary measure is configural reliance (CR), the scramble loss as a share of above-chance accuracy on the grid control; the absolute loss is reported alongside.

```latex
\mathrm{CR} = \frac{A_{\mathrm{grid}} - A_{\mathrm{scr}}}{A_{\mathrm{grid}} - 1/16}
```

Here A\_grid is 16-way accuracy on the 12 px grid control and A\_scr on the scrambles, averaged over an object's scrambles first. Scrambles that change less than 10% of the object's layout (1 minus IoU) are excluded. CR is 0 when scrambling costs nothing and 1 when it removes all above-chance accuracy. It removes the ceiling that made absolute losses depend on control accuracy (BagNet-33 and CLIP lose similar shares, 0.17 and 0.18, but very different absolute amounts).

CR became primary after the Stage 2 calibration run; the pre-specified absolute measure stays as a secondary outcome. A lesion's effect is the change it causes, ΔCR = CR(lesioned) minus CR(intact).

CR needs headroom to be meaningful. If a lesion leaves the lower 95% bound of A\_grid below 0.125 (twice chance, the floor used since Stage 1), CR is not computed for that lesion. It is reported as floor and counts as not supporting any hypothesis that depends on it.

| Measure | Definition | Used in |
| --- | --- | --- |
| Absolute scramble loss | A\_grid minus A\_scr | H1 to H3, secondary |
| Damage | Drop in grid-control accuracy caused by a lesion | Matching control lesions |
| Hue decoding | Linear-probe accuracy for the object's dominant hue (6 bins, pixels with saturation above 0.2) | H4 |
| Local category decoding | Linear-probe accuracy for category from scrambled-image features | H4 |
| Error locality | Jensen-Shannon distance between the model's error distribution and BagNet-33's, over off-diagonal confusions | H5 |
| Line-drawing advantage | Edge-set minus silhouette-set accuracy, paired over the same 160 images | H6 |
| Overlap cost | Both-objects-correct rate of a multi-label probe, side by side minus small overlap | H7 |

All accuracies are 16-way and category-balanced. Frankenstein losses are computed the same way and reported descriptively.

## Unit localisation

Channels are selected by their causal effect on configural reliance, not by response selectivity, because selectivity is a poor guide to what a unit does. Single-unit class selectivity predicts ablation damage poorly ([Morcos et al., 2018](https://arxiv.org/abs/1803.06959v3)) and is neither necessary nor sufficient for performance ([Leavitt & Morcos, 2020](https://arxiv.org/abs/2003.01262v2)). Dobs et al. found ranking filters by their effect on the loss more informative than ranking by selectivity, and lesioned the top 20% ([Dobs et al., 2022](https://dspace.mit.edu/handle/1721.1/148825)).

**Unit.** A unit is one channel (feature map) at a site, ablated across all spatial positions, as for convolutional networks in Morcos et al. In CORnet-S the ablation acts on an area's output, so it removes that channel's signal to the next area after recurrence.

**Configural importance (primary score).** Each channel is zero-ablated alone and its effect measured on the localizer half, textured stimuli. P is the mean probability of the true category, which varies more smoothly than accuracy.

```latex
I_{\mathrm{conf}}(c) = \left[P_{\mathrm{grid}} - P_{\mathrm{grid}}^{(-c)}\right] - \left[P_{\mathrm{scr}} - P_{\mathrm{scr}}^{(-c)}\right]
```

A positive score means the channel supports intact objects more than scrambled ones. The local score is the channel's support for scrambled objects, P\_scr minus P\_scr without c.

**Amended (A1, 15 October 2026).** The calibration pilot showed that this absolute difference favours channels that matter for every image, because intact objects start from a higher probability and so lose more in absolute terms. The primary score is now the channel's effect on configural reliance itself, computed on true-category probabilities: CR\_p = 1 minus (P\_scr minus 1/16) / (P\_grid minus 1/16), and I\_CR(c) = CR\_p(intact) minus CR\_p(with c ablated). Significance uses a one-sided, category-stratified bootstrap over objects (10,000 resamples) instead of the t-test, with the same Benjamini-Hochberg control and 1 to 40% bounds. Every rule that ranked by I\_conf now ranks by I\_CR. Full text: amendment\_01.md in the preregistration repository.

**Reliability.** Scores are recomputed on the calibration test half. A site enters the lesion analyses only if the split-half Spearman correlation of channel scores, Spearman-Brown corrected, reaches 0.7. Failing sites are reported, not lesioned.

**Sets.** At each site, with n channels:

- **Configural set:** channels whose configural importance is reliably above 0, by a one-sided test across localizer objects with Benjamini-Hochberg FDR at q = 0.05, the same logic Prince et al. use for selective units. The set is bounded to 1 to 40% of the site's channels (the dose-response range); outside that range the top 1% or 40% is taken. Dobs et al. give no rationale for their 20% and test no other size, and no biological lesion fraction maps onto channels: patients' lesions remove contiguous tissue, not a functionally chosen subset. The data therefore set the size.
- **Local set:** from channels with configural importance at or below the site median, the 20% with the highest local score.
- **Selective set (secondary):** channels whose spatially averaged response is higher for grid controls than scrambles, by a paired t-test across objects with Benjamini-Hochberg FDR at q = 0.05, as for category-selective units in [Prince, Alvarez & Konkle (2024)](https://www.biorxiv.org/content/10.1101/2023.08.04.551888.full.pdf). Its overlap with the configural set (Jaccard index) tests whether selectivity and importance agree here.

**Primary site.** For each model, the reliable site whose configural set has the largest summed configural importance on the localizer half. The rule is fixed now and applied to calibration data only; all other reliable sites are reported as secondary.

## Lesion protocol

The primary lesion zeroes the configural set (sized by the data, see Unit localisation) at each model's primary site and is compared with random lesions matched for size and for damage. All sets are fixed on the calibration set before the hold-out set is used.

**Ablation.** Channels are set to zero, the method used by Morcos et al., Dobs et al. and Prince et al. Every total ablation moves a network off its training distribution ([Li & Janson, Optimal ablation for interpretability](https://arxiv.org/pdf/2409.09951)), so all primary contrasts are repeated with mean ablation (each channel replaced by its mean over calibration images). A result is called robust only if both methods agree in direction and significance.

| Condition | How the channels are chosen | Role |
| --- | --- | --- |
| Configural | FDR-significant (1 to 40% of channels) by configural importance | Target |
| Random, size-matched | 100 random sets of the same size and site, seeded | H2 control, after Prince et al. |
| Random, damage-matched | From up to 10,000 random sets, the 100 whose damage is within 0.02 of the configural set's (else the 100 closest) | H2 control for total harm |
| Local | Top channels by local score, the set size chosen so its damage is closest to the configural set's | H3 contrast |
| Selective | FDR-selected configural-selective channels | Secondary |

Damage is measured on the calibration localizer half, so matching never looks at hold-out results.

**Secondary lesions.**

- Dose-response at the primary site: 1, 2, 5, 10, 20 and 40% of channels, configural against 20 size-matched random sets per size.
- Every other reliable site, its configural set defined by the same rule, against 20 size-matched random sets.

**Exploratory.** Progressive lesioning with brief retraining, after [Shah, Tran & Yamins (2024)](https://2024.ccneuro.org/poster/?id=268), tests whether configural reliance recovers or is replaced by local strategies. It is defined after the confirmatory analyses and reported as exploratory.

The networks are deterministic in evaluation mode; the only randomness is in the control sets, which use fixed seeds.

## Agnosia signature tests

The secondary hypotheses test whether the lesioned network shows Dr P's profile: local evidence kept, wholes lost, errors driven by parts.

**Probes.** Read-outs are linear (L2-regularised logistic regression) on penultimate-layer features, because linear probes are the most selective in the comparison by [Hewitt & Liang (2019)](https://arxiv.org/pdf/1909.03368). Probes are retrained for each lesion condition on calibration features, with the regularisation chosen by 5-fold cross-validation grouped by object, then tested on hold-out features. Each probe is paired with a control task, the same probe trained on labels shuffled at the object level; probe selectivity (task minus control accuracy) is reported.

- **H4, local evidence.** Two probes: dominant hue, and category trained and tested on scrambled images. Supported if each drop after the configural lesion exceeds the mean drop after damage-matched random lesions by less than 0.05 (upper confidence bound below 0.05). The same category probe on grid controls is reported to show the whole-object loss at the representation level.
- **H5, local errors.** On hold-out grid controls, the off-diagonal confusion distribution is compared with BagNet-33's by Jensen-Shannon distance. Supported if the distance falls more after the configural lesion than after damage-matched random lesions.
- **H6, line-drawing advantage.** Edge-set minus silhouette-set accuracy over the same 160 public images. Supported if it falls more after the configural lesion than after damage-matched random lesions. The reference sets are public, so this is the one test not run on hold-out images; the unlesioned values were seen in Stage 2.
- **H7, overlapping figures.** A 16-output multi-label linear probe, trained on calibration outlines and pairs, counts a trial correct when its top two outputs are the two objects. Supported if the overlap cost rises more after the configural lesion than after damage-matched random lesions. Patient S.M. was at chance on overlapping figures but better with the same shapes apart (Behrmann & Williams, 2007).

**Exploratory.** Navon letters (global and local letter probes), Frankenstein effects after lesions, and per-category profiles are reported without hypotheses.

## Statistical analysis

Every test uses a category-stratified cluster bootstrap over objects (10,000 resamples), category-balanced means, and Holm correction within each hypothesis family.

**Resampling.** The object is the independent unit; its variants (scrambles, Frankenstein halves) are averaged first. Objects are resampled within category, so every resample keeps the category balance. Lesion conditions are resampled jointly, so contrasts between them are paired. CR is a ratio of means and is recomputed in each resample; intervals are percentile intervals.

**Decision rules.**

| Hypothesis | Statistic | Supported if |
| --- | --- | --- |
| H1 | CR(model) minus CR(BagNet-33) | Lower 95% bound above 0 |
| H2 | ΔCR(configural) minus mean ΔCR(random), for size-matched and damage-matched controls | Both upper bounds below 0, and ΔCR(configural) below the 5th percentile of the 100 damage-matched random lesions |
| H3 | ΔCR(configural) minus ΔCR(local) | Upper bound below 0 |
| H4 | Probe drop, configural minus damage-matched random | Upper bound below 0.05 (non-inferiority) |
| H5 to H7 | Change after configural minus change after damage-matched random | Bound excludes 0 in the predicted direction |

**Multiplicity.** The primary family is H1 to H3 in two models (6 tests); the secondary family is H4 to H7 in two models (8 tests). Two-sided bootstrap p-values are Holm-corrected within each family at a family-wise 0.05.

**Margin.** The 0.05 accuracy margin is the same bound that defined negligible local effects when the stimuli were validated, so "negligible" means one thing throughout. Equivalence and non-inferiority follow [Lakens (2017)](https://doi.org/10.1177/1948550617697177), with the smallest effect of interest fixed in raw accuracy units.

**Scope of claims.** The primary model decides each headline result. A result is called general only if both models support it.

**Sensitivity analyses.** Mean ablation, the absolute scramble loss, source-weighted means, and all secondary sites. They are reported in full whether or not they agree with the primary analysis.

## Failure and reporting

Every hypothesis is reported whatever its outcome, and each failure pattern already has a fixed interpretation.

| Outcome | Interpretation | How it is reported |
| --- | --- | --- |
| H1 fails for a model | Its configural reliance did not replicate beyond calibration | Lesion results for that model are reported as conditional on an unreplicated effect |
| No site passes reliability | Configural importance cannot be measured stably per channel | Reported as a negative result; no lesion claim for that model |
| H2 fails | Configural reliance is not concentrated in a sparse channel set at the tested sites | Reported as evidence for a distributed code, with the dose-response curve |
| H2 holds, H3 fails | The lesion harms recognition generally rather than integration specifically | No dissociation is claimed |
| Zero and mean ablation disagree | The effect depends on the ablation method | Called method-dependent, not robust |
| Some of H4 to H7 fail | The lesioned profile matches the syndrome only in part | The matching and non-matching features are listed side by side |

**Exploratory analyses** are labelled as such and carry no confirmatory claim: progressive lesioning with retraining, unit characterisation (receptive fields, attribution maps), Navon probes, Frankenstein effects after lesions, per-category results, and the seven unselected models.

## Design history and deviations

Six design changes were made before this plan, all on calibration data and all listed in the paper's Methods. Changes after the freeze are appended here, newest first; amendment A1 is the only one.

| Date | Change | Reason | Data seen before the change |
| --- | --- | --- | --- |
| 15 Oct 2026 (after freeze) | Amendment A1: unit score changed from absolute probability differences to the channel's effect on configural reliance (I\_CR), tested by object bootstrap; the only revision to unit selection | The absolute score favoured generally important channels; the configural lesion raised CR in the pilot (ResNet-50 SWSL 0.542 to 0.690) and removed all recognition in CORnet-S | Stage 3 calibration scores, sets and pilot; hold-out untouched |
| 15 Oct 2026 | CLIP reloaded with its matching activation (QuickGELU) and Stage 2 rerun | Loader bug; selection unchanged | Stage 2 calibration results |
| 15 Oct 2026 | Configural reliance (CR) made primary; absolute loss kept as secondary | Absolute losses are capped by control accuracy | Stage 2 calibration results |
| 15 Oct 2026 | Four candidates added (adversarial and SWSL ResNet-50, ConvNeXt-T, CLIP) | Standard reviewer questions on robustness, data scale and modern models | None for these models |
| 15 Oct 2026 | Hold-out set generated and locked; knife has no hold-out objects | Independent confirmation; calibration used every eligible knife | BagNet controls only on hold-out |
| 15 Oct 2026 | Grid gap widened from 6 to 12 px by a pre-specified rule | BagNet-33 scramble drop 0.054 exceeded the 0.05 bound at 6 px | Stage 1 calibration results |
| 15 Oct 2026 | Textured renderings added to the silhouettes | Local-only controls were at chance on silhouettes | Stage 1 calibration results |

## References

- Baker, N., & Elder, J. H. (2022). Deep learning models fail to capture the configural nature of human shape perception. *iScience*, 25(9), 104913. [Link](https://doaj.org/article/33ba76d856ce4f14ac57cd060a6aadb8)
- Behrmann, M., & Williams, P. (2007). Impairments in part-whole representations of objects in two cases of integrative visual agnosia. *Cognitive Neuropsychology*, 24(7), 701 to 730. [PDF](https://www.cmu.edu/dietrich/behrmannlab/Publications/Impairments%20in%20part%20whole%20representations%20of%20objects%20in%20two%20cases%20of%20integrative%20visual%20agnosia.pdf)
- Dobs, K., Martinez, J., Kell, A. J. E., & Kanwisher, N. (2022). Brain-like functional specialization emerges spontaneously in deep neural networks. *Science Advances*, 8(11). [Link](https://dspace.mit.edu/handle/1721.1/148825)
- Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. arXiv:1909.03368. [Link](https://arxiv.org/pdf/1909.03368)
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*. [DOI](https://doi.org/10.1177/1948550617697177)
- Lawson, R., & Humphreys, G. W. (1999). The effects of view in depth on the identification of line drawings and silhouettes of familiar objects: Normality and pathology. *Visual Cognition*, 6, 165 to 195. [Link](https://www.liverpool.ac.uk/~rlawson/HTML_TXT_Files/L&H-VisCog-1999.html)
- Leavitt, M. L., & Morcos, A. (2020). Selectivity considered harmful: Evaluating the causal impact of class selectivity in DNNs. arXiv:2003.01262. [Link](https://arxiv.org/abs/2003.01262v2)
- Li, M., & Janson, L. Optimal ablation for interpretability. arXiv:2409.09951. [Link](https://arxiv.org/pdf/2409.09951)
- Morcos, A. S., Barrett, D. G. T., Rabinowitz, N. C., & Botvinick, M. (2018). On the importance of single directions for generalization. arXiv:1803.06959. [Link](https://arxiv.org/abs/1803.06959v3)
- Prince, J. S., Alvarez, G. A., & Konkle, T. Contrastive learning explains the emergence and function of visual category-selective regions. bioRxiv 2023.08.04.551888. [Link](https://www.biorxiv.org/content/10.1101/2023.08.04.551888.full.pdf)
- Riddoch, M. J., & Humphreys, G. W. (1987). A case of integrative visual agnosia. *Brain*. [Summary](https://en.wikipedia.org/wiki/Integrative_agnosia)
- Riddoch, M. J., Humphreys, G. W., Akhtar, N., Allen, H., & Bracewell, R. M. (2008). A tale of two agnosias: Distinctions between form and integrative agnosia. *Cognitive Neuropsychology*, 25(1), 56 to 92. [Link](<https://research.bangor.ac.uk/portal/en/researchoutputs/a-tale-of-two-agnosias-distinctions-between-form-and-integrative-agnosia(2ebd34b9-5fc0-49bc-9bd4-f46a8b9a5ecb).html>)
- Sacks, O. (1985). *The Man Who Mistook His Wife for a Hat*. Summit Books.
- Shah, Y., Tran, K., & Yamins, D. (2024). Modeling localized synaptic degeneration and neural plasticity in visual cortex. Cognitive Computational Neuroscience. [Link](https://2024.ccneuro.org/poster/?id=268)
