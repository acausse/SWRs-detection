# Methods-to-code guide and fidelity notes

## Primary source

Causse, A. A., Curot, J., Lopes-dos-Santos, V., et al. (2026),
*A learning-evoked slow-oscillatory architecture paces population activity for
offline reactivation across the human medial temporal lobe*, **Neuron 114,
3786-3798**. https://doi.org/10.1016/j.neuron.2026.05.004

**Source section:** STAR Methods, **"Detection of hippocampal ripples"**,
with subsections:

- "Initial ripple detection using overnight sleep recordings"
- "Dimensional embedding and template generation"
- "Loose detection of putative ripple events"
- "Template matching and white-matter control"

The second half of this methodology is what this repository executes. The
sleep-based training pipeline is considered already complete, and its output
(the validated templates) is a *required user input*.

## Research workflow versus repository scope

```text
PAPER'S TRAINING STAGE (NOT IMPLEMENTED HERE)
Overnight slow-wave-sleep LFPs
 -> initial 60-180 Hz ripple detection with WM control
 -> candidate waveform inspection / embedding / clustering
 -> 14 validated representative templates
                            |
                            | user supplies existing templates
                            v
GENERALIZED STANDALONE DETECTOR (IMPLEMENTED)
Probe LFP (lfp)          White-matter LFP (ref)
        |                        |
        | 60-180 Hz bandpass     |
        | analytic envelope     |
        | smoothing             |
        | adaptive threshold    |
        | artifact exclusions   |
        v                        |
   loose events                  |
        |                        |
        +---- matching-band -----+
             local peak in
             EACH signal
                  |
       match templates separately
                  |
     best_scores, best_scores_ref
                  |
     accept: probe >= 0.85 AND WM < 0.85
                  |
     event arrays + full candidate score file
```

## Step A - permissive candidate detection

**Paper:** "Loose detection of putative ripple events".

- Search in the 60-180 Hz ripple range.
- Calculate the analytic amplitude envelope, smooth with **80 ms width** and
  **40 ms Gaussian SD**.
- Choose an adaptive threshold targeting **20-30 candidate events/minute**.
- Require duration **>=20 ms** and separation **>=150 ms**.
- Exclude candidate intervals overlapping interictal epileptiform discharge
  (IED) windows. The standalone function receives such windows but does not
  detect IEDs independently.
- Calculate a narrowband 80-Hz wavelet signal to identify phase-aligned peaks
  and troughs (the matched waveforms are subsequently aligned to their own
  ripple-band local maxima).

**Implemented by:** `detect_swrs`, `bandPass`, `get_smooth`,
`detect_burst_events`, `detect_burst_events_adaptive_thresh`,
`exclude_bad_windows`, `get_wv_hilbert`, `get_peakClosest2Time`.

**Not identical to the original:** The active old `MODE='hfb'` uses precomputed
high-frequency broadband (`bb.load_hfb`) that is not a universal LFP input.
The standalone defaults to the 60-180 Hz Hilbert envelope of `lfp`, then
z-scores and smooths it. A user may choose `envelope_normalization='raw'`.
`N=100` and `sigma=50` apply at 1250 Hz and scale with `sr`.

The adaptive target is a *candidate count*, **not** a guarantee about the
number of true or accepted ripples. The standalone algorithm records the
threshold it used and warns when the target interval was not reached.

## Step B - fixed-template comparison in two contacts

**Paper:** "Template matching and white-matter control".

- For each candidate event, align local waveforms to a peak in the ripple-band
  signal **separately** for probe and white-matter control.
- Compare every waveform against every supplied template, choosing the
  highest similarity score per contact.
- Select events that are morphologically consistent with validated ripple
  templates **and** not matched at the white-matter control contact.

**Implemented by:** `getPeaksWithinWindows`, `extract_snippets`,
`match_cosine_or_pearson`, and `detect_swrs`.

The default similarity (`lag_samples=0`) is cosine similarity after
mean-centering each waveform; equivalently, it is Pearson correlation.
The maximum score for one candidate may come from different templates in
probe and control, as it does in the separately scored original files.

**Exact original software classification:**

```python
idx_orig_only = (best_scores >= thresh) & (best_scores_ref < thresh)
idx_nonmatch = (best_scores < thresh)
idx_orig_and_ref = (best_scores >= thresh) & (best_scores_ref >= thresh)
```

The method's `thresh=0.85` was chosen using the white-matter match-score
null distribution in the paper's data (95th percentile). This threshold's
external validity should be tested on new acquisition systems and templates.

## Step C - persist candidates and filter without recomputation

**Implementation convenience, not a new methodological stage.**

`save_swr_results` stores **all matchable candidates** and scores, not just
accepted events. `load_swr_new` then reproduces the threshold selection logic
of original `ac_baseObjects.bblocks.load_swr_new`. With the default
`RETURN_BAD_INSTEAD=False`, it returns accepted probe-only events. With
`RETURN_BAD_INSTEAD=True`, it returns probe **nonmatches**; simultaneous
probe-and-WM matches remain excluded from both returned subsets.

The new loader uses a single results file rather than the original
contact-specific filename plus `_wmREF-...` companion file.

## Important boundaries between paper and code

| Topic | Paper | Supplied original script | Standalone script |
| --- | --- | --- | --- |
| Template generation | Sleep verification, Isomap, K-means, manual selection, 14 templates | Has a separate optional training branch | **Never trains**; user supplies file |
| Candidate frequency band | 60-180 Hz | `rippleBand=[60,180]` in original candidate script | 60-180 Hz default |
| Matching band | Describes matching ripple-band waveforms | `ripple_band=[50,250]` in clean-template script | **50-250 Hz default**, editable |
| Smoothing | width 80 ms, SD 40 ms | `N=100`, `sigma=50` at 1250 Hz, active HFB branch | same nominal window, using Hilbert envelope |
| Candidate threshold | 20-30/min, duration >=20 ms, separation >=150 ms | adaptive binary search | adaptive search, closest-count fallback |
| IEDs | Excluded | `bb.load_IEDs_macro` | supply `bad_windows`; no IED detector bundled |
| Template snippet window | Sleep-training section describes 25-ms snippets | Matching extracts +/-25 ms (about 50 ms total) | Derives matching duration from template length |
| Template score cutoff | text says >0.85, based on white-matter null | implements `>=0.85` for probe and `<0.85` for WM | **same software convention** (`>=`, `<`) |
| Broadband ripple-triggered averages | +/-250 ms | `taLen_s=0.25` in shown plotting script (250 ms total) | 250 ms total, configurable, **not** a paper average |
| Spectrograms | downstream 10-200 Hz analysis | `vbf.wvSpect` / `snipSpect` | **not computed** |
| Numerical filtering | methodological description | `butter` with `(b,a)` and `filtfilt` | zero-phase Butterworth SOS; near-equivalent, not bit-identical |

## Interpreting new applications

Use an anatomically credible reference trace. Record the acquisition and
processing montage, template library version, sampling frequencies, artifact
masking, detection settings, and software commit/version. Assess the rate of
accepted detections, event morphology, event frequency, temporal distribution,
reference-contact matching and agreement with independent expert review.
Synthetic tests verify code behavior, **not** physiological validity.
