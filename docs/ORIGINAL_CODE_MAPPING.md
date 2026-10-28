# Crosswalk: generalized SWR detector and original laboratory scripts

**Audience:** Researchers who have access to **both** the original analysis
repository and this standalone implementation. The objective is to find the
original functions or script statements corresponding to each generalized
step, without depending on `bblocks` or the laboratory file layout.

This mapping is based on the following supplied files (names as originally
provided):

- `ac_run_detect_ffaBursts_macro.py`
- `ac_run_clean_swrs_templates.py`
- `ac_functions_swrs_clean_templates.py`
- `ac_baseFunctions.py`
- `ac_baseObjects.py`
- `ac_vlds_swr_copy.py`
- `vBaseFunctions3.py`

The new implementation is **one file**:
`general_swr_detection_with_loader.py`. Refer to the function names as anchors
when line numbers differ across code versions. Line numbers below refer to the
supplied copies.

## End-to-end sequence: old versus new

| Stage | Original file and location | New file / function | Notes |
| --- | --- | --- | --- |
| Recording metadata and LFP load | `ac_run_detect_ffaBursts_macro.py` ~101-142; `bb=acbo.bblocks()`, `bb.readAttribute(...)`, `bb.load_eeg(...)`; implementation in `ac_baseObjects.py` | **USER INPUTS**: supply `lfp`, `ref`, `sr` to `detect_swrs` | All dataset/session/channel selection is external to detector. |
| White-matter control | Old `ac_baseObjects.bblocks.load_swr_new` ~3443 onward discovers `_wmREF-...` file for the channel | `detect_swrs(lfp, ref, ...)` | The white-matter trace is provided directly and must have matching timestamps. |
| Artifact/IED exclusion | `ac_run_detect_ffaBursts_macro.py` ~132, 153-156 calls `bb.load_IEDs_macro`; `ac_baseFunctions.py` ~4149 excludes overlaps | `prepare_bad_windows`, `exclude_bad_windows` | Pass `bad_windows` as an `(n,2)` sample-index array. |
| Candidate-band filtering | `ac_run_detect_ffaBursts_macro.py` ~144-151 uses `ac_swr.bandPass` / `vbf.runHilbert`; `ac_vlds_swr_copy.bandPass` ~421 | `bandPass`, `scipy.signal.hilbert` in `detect_swrs` | Default `rippleBand=[60,180]` Hz. |
| HFB optional branch | `ac_run_detect_ffaBursts_macro.py` ~144-146 calls `bb.load_hfb`, z-scores and smooths it | `envelope_normalization` option | No precomputed `bb.load_hfb`; standalone generalization uses a Hilbert envelope from `lfp`. |
| Envelope smoothing | `ac_run_detect_ffaBursts_macro.py` ~85-87, 146; `ac_baseFunctions.get_smooth` ~3045 | `get_smooth` | `N=100`, `sigma=50` at 1250 Hz; divide by `N` if selected. |
| Adaptive candidate detection | `ac_run_detect_ffaBursts_macro.py` ~168-174 calls `ac.detect_burst_events_adaptive_thresh`; definitions in `ac_baseFunctions.py` ~4101, 4149 | `detect_burst_events`, `detect_burst_events_adaptive_thresh` | 20-30 candidates/min, duration >=20 ms, separation >=150 ms. |
| Edge exclusion | `ac_run_detect_ffaBursts_macro.py` ~176-179 | edge mask in `detect_swrs` | Default 0.5 s margin; new code also safeguards matching and broadband snippet borders. |
| Narrowband realignment | `ac_run_detect_ffaBursts_macro.py` ~180-182 calls `ac.get_wv_hilbert`, `ac.get_peakClosest2Time`; definitions in `ac_baseFunctions.py` ~5025 and ~2138 | `get_wv_hilbert`, `get_peakClosest2Time` | Calculates 80-Hz troughs/peaks; used as metadata in new code, as matching uses local peaks. |
| Loading fixed templates | `ac_run_clean_swrs_templates.py` ~111-126 reads `templates_v1_*.pkl`; new script does **not** use its `else` training branch ~128-240 | `load_templates` | Accepts pre-trained `.pkl`, `.npz`, `.npy`. |
| Extract matching-band LFP | `ac_run_clean_swrs_templates.py` ~273-286 uses `vbf.bandpass`, `vbf.getPeaksWithinWindows`; definitions in `vBaseFunctions3.py` ~401 and ~1364 | `bandPass`, `getPeaksWithinWindows`, `extract_snippets` | `ripple_band=[50,250]` Hz, local peaks within candidate windows. |
| Cosine/Pearson-like matching | `ac_run_clean_swrs_templates.py` ~294-295 calls `acsw.match_cosine_or_pearson`; definition in `ac_functions_swrs_clean_templates.py` ~220 | `match_cosine_or_pearson` | Default `lag_samples=0`, maximum score over all templates. |
| Comparison to WM | `ac_baseObjects.bblocks.load_swr_new` ~3500-3517 loads separate probe and `_wmREF-...` scores | second matching pass in `detect_swrs` | Both signals are scored separately using the same fixed templates. |
| Three match categories | `ac_functions_swrs_clean_templates.categorize_event_matches` ~919-961 | `categorize_event_matches` | `idx_orig_only`, `idx_nonmatch`, `idx_orig_and_ref`. |
| Final accepted ripple subset | `ac_baseObjects.bblocks.load_swr_new` ~3515-3540 | `detect_swrs` -> `good_mask`; `load_swr_new` | Accept if `best_scores >= thresh` and `best_scores_ref < thresh`. |
| LFP-triggered snippets | `ac_run_clean_swrs_templates.py` ~288-292 calls `vbf.triggeredAverage`; definition ~855 | `extract_snippets` for `snipLFPs_candidates` | Keeps broadband snippets; no spectrogram calculation in generalized code. |
| Save / reload | `ac_run_detect_ffaBursts_macro.py` ~198-227, `ac_run_clean_swrs_templates.py` ~318-344, `ac_baseObjects.load_swr_new` | `save_swr_results`, `save_candidates_csv`, `load_swr_new` | Saved *candidate* dictionary is rethresholdable. |

## Where the two main passes are located

### 1. Loose detection (original candidate detector)

Open `ac_run_detect_ffaBursts_macro.py`, then look for:

```python
MODE=['hfb', 'rip'][0]
rippleBand=[60,180]
N=100
sigma=50
min_events_mn=20
max_events_mn=30
min_separation_ms=150
min_duration_ms=20
```

In the channel loop, the original calls:

```python
ac.detect_burst_events_adaptive_thresh(band_sm, bad_windows, sr=bb.sr, ...)
```

The new `detect_swrs` computes `band_sm` directly from the supplied `lfp`
using a bandpass-filtered Hilbert envelope and then calls the standalone
`detect_burst_events_adaptive_thresh`. The new file has no `bb` object.

**Important distinction:** In the supplied original script, `MODE` is
actually selected as `'hfb'` (the first value). That branch loads precomputed
HFB (`bb.load_hfb`), then z-scores and smooths it. The new file's default
builds the *ripple-band Hilbert amplitude* from `lfp`, z-scores it and smooths
it. These are related but not always numerically equivalent operations.

### 2. Template matching (original clean-template script)

Open `ac_run_clean_swrs_templates.py` around lines 273-295:

```python
ripple_band = [50, 250]
window_size = int(0.025 * 1250)
event_intervals = events[:, [0, 2]]
ripple_band_lfp = vbf.bandpass(lfp, ripple_band[0], ripple_band[1])
peaks = vbf.getPeaksWithinWindows(ripple_band_lfp, event_intervals)
snippets = np.array([
    ripple_band_lfp[timestamp-window_size:timestamp+window_size]
    for timestamp in peaks
])
best_scores, best_tidx, best_lag, best_full = \
    acsw.match_cosine_or_pearson(templates, snippets)
```

The new `detect_swrs` performs the same conceptual operations for **both**
`lfp` and `ref`, using the same candidate intervals, but it derives matching
snippet length from `templates.shape[1]` to guarantee consistent dimensions.

**Window-length distinction:** `window_size=int(0.025*1250)` in the old code
is a *half-window*, so the resulting snippet is around 50 ms long (62 samples
at 1250 Hz), despite the paper's 25-ms description of the *sleep template
extraction* step. The standalone loader trusts the actual length of your
saved templates rather than assuming 25 or 50 ms.

### 3. Filtering for later analyses (original `bb.load_swr_new`)

Open `ac_baseObjects.py`, method `bblocks.load_swr_new` starting around line
3443. It constructs a probe path, locates a matching `_wmREF-...` pickle,
loads both dictionaries and invokes:

```python
idx_orig_only, idx_nonmatch, idx_orig_and_ref = \
    acsw.categorize_event_matches(
        dic_actual['best_scores'], dic_ref['best_scores'], thresh=thresh
    )
```

It returns original-only events by default; `RETURN_BAD_INSTEAD=True` returns
**probe nonmatches**, not events matching both contacts. The generalized
`load_swr_new(fname, ...)` replicates the classification and selection but
loads a **single new-format result pickle**, because the two signals were
already scored in the same `detect_swrs` call.

Original call:

```python
swrDic = bb.load_swr_new(bb.seshLabs[ses], chreg, thresh=0.85)
```

Standalone call:

```python
swrDic = load_swr_new("results/my_recording_candidates.pkl", thresh=0.85)
```

## Original components deliberately omitted

- `ac_baseObjects.bblocks`, channel label lookup, anatomical labels, file
  naming, session selection, macrocontact data access and cache directories.
- The template training branch of `ac_run_clean_swrs_templates.py`
  (`swrs_summary_df.pkl` -> bootstrap averages -> Isomap -> K-means).
- The overnight-sleep initial template derivation and manual template curation
  described in the paper.
- `ccwBaseFunctions` and plotting convenience utilities.
- Wavelet spectrogram construction (`vbf.wvSpect`) and most figures; the
  original `snipSpect` and `freqs_spectro` fields are not produced by the
  standalone detector.

## Differences which matter for numerical reproducibility

1. `get_smooth`: standalone uses an FFT convolution but retains the original
   `/N` and crop conventions; floating-point differences are possible.
2. `bandPass`: standalone uses stable second-order sections (SOS) and zero
   phase; original filters use direct-form Butterworth `b, a` coefficients.
3. `detect_burst_events_adaptive_thresh`: original returns the *last tried*
   threshold when no threshold meets the event-count interval. Standalone
   returns the *closest tested* count. This is a deliberate improvement.
4. `get_wv_hilbert` and local extrema: implementations are analogous, not
   guaranteed sample-for-sample identical. Stored peak alignments may differ.
5. `lag_samples>0`: standalone re-centers/normalizes each overlap to avoid the
   nonzero-lag bias in the old implementation. The main original call uses 0.
6. `envelope_normalization='zscore'`: new default applied to Hilbert envelope;
   differs from the old active HFB branch and unnormalized ripple branch.
7. `bad_windows`: standalone interval semantics are `[start, end)` with explicit
   overlap exclusion; check conversion from your original artifact files.
8. `taLen_s=0.25`: a *total* 250-ms broadband snippet. The paper's
   ripple-triggered averages use **+/- 250 ms**, a separate downstream step.

The scientific criterion is preserved, but identical candidate timestamps or
counts are **not guaranteed**. To benchmark reproducibility, feed the same
`lfp`, `ref`, `sr`, templates and excluded intervals into both implementations
and compare candidates, peaks, similarity scores and final accepted events.

See also [Methods notes](METHODS_AND_IMPLEMENTATION.md) and
[Output schema](OUTPUT_SCHEMA.md).
