# Quick crosswalk: original code to standalone SWR detector

This short guide is for a reader who has **both** your original scripts and the
new `general_swr_detection_with_loader.py`. Locate functions by name; line
numbers refer to the supplied snapshots and may differ in other versions.

## End-to-end comparison

| Operation | Original location | Standalone location |
| --- | --- | --- |
| Load selected probe LFP | `ac_run_detect_ffaBursts_macro.py`: `bb.load_eeg(...)` / `ac_baseObjects.bblocks` | **USER INPUTS**, supply 1D `lfp` and `sr` |
| Load/reference WM signal | `ac_baseObjects.bblocks.load_swr_new`: companion `_wmREF-...` result file | **USER INPUTS**, supply 1D `ref` on same timebase |
| High-frequency envelope | `ac_run_detect_ffaBursts_macro.py`: `MODE`, `ac_swr.bandPass`, `vbf.runHilbert` | `detect_swrs`, `bandPass`, SciPy Hilbert |
| Smooth envelope | `ac_baseFunctions.get_smooth` | `get_smooth` |
| Detect putative events | `ac_baseFunctions.detect_burst_events`, `detect_burst_events_adaptive_thresh` | Functions of same names |
| Exclude IED periods | `bb.load_IEDs_macro`, event exclusion inside original adaptive detector | `bad_windows`, `exclude_bad_windows` |
| 80-Hz alignment | `ac_baseFunctions.get_wv_hilbert`, `get_peakClosest2Time` | Functions of same names |
| Load already-built templates | `ac_run_clean_swrs_templates.py`: saved `templates_v1_*.pkl`, `templates` | `load_templates(templates_path, sr_templates, sr)` |
| Find peaks and matching snippets | `ac_run_clean_swrs_templates.py`: `vbf.bandpass`, `vbf.getPeaksWithinWindows` | `bandPass`, `getPeaksWithinWindows`, `extract_snippets` |
| Score morphological matches | `ac_functions_swrs_clean_templates.match_cosine_or_pearson` | Function of same name; probe AND WM |
| Select reliable events | `ac_functions_swrs_clean_templates.categorize_event_matches` and `ac_baseObjects.bblocks.load_swr_new` | `categorize_event_matches`, `load_swr_new` |
| Save candidates | `ac.saveAsPickle`, separate subject/contact files | `save_swr_results`, single per-recording candidate pickle |

## Equivalent selection call

```python
# ORIGINAL: session/contact-specific loader
swrDic = bb.load_swr_new(bb.seshLabs[ses], chreg, thresh=0.85)

# NEW: saved output of detect_swrs for a single pair of LFPs
swrDic = load_swr_new("results/my_recording_candidates.pkl", thresh=0.85)
```

Both use:

```python
good = (best_scores >= 0.85) & (best_scores_ref < 0.85)
```

`RETURN_BAD_INSTEAD=True` selects **probe nonmatches**, *not* events matching
both signals. A threshold can be reapplied to the saved candidate pickle
without running the detector again.

## What is intentionally different

The new script does not depend on `bblocks`, laboratory file paths, anatomical
labels, or any template training procedure (`swrs_summary_df.pkl`, Isomap,
K-means). It uses a generic Hilbert envelope derived from `lfp`, unlike the
active original `MODE='hfb'` branch with precomputed `bb.load_hfb`.

Template waveform length comes from the loaded file, and the new script does
not generate `snipSpect`. Other numerical differences in filtering and
alignment are documented in
[ORIGINAL_CODE_MAPPING.md](ORIGINAL_CODE_MAPPING.md).

**Paper:** Causse et al. (2026), *Neuron*, STAR Methods, "Detection of
hippocampal ripples", especially "Loose detection of putative ripple events"
and "Template matching and white-matter control":
https://doi.org/10.1016/j.neuron.2026.05.004
