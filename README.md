# Generalized hippocampal SWR detection using fixed ripple templates

A standalone Python implementation for **one probe LFP** (`lfp`) and **one
synchronized white-matter control LFP** (`ref`). It detects loose high-frequency
candidate events, compares their waveforms against **pre-existing validated
ripple templates**, and accepts events that match in the probe but not in the
control. It also saves *all* candidate scores so the acceptance threshold can
be changed later without reprocessing the original LFPs.

The primary algorithm is based on the following paper:

**Causse, A. A., Curot, J., Lopes-dos-Santos, V., et al. (2026).**
*A learning-evoked slow-oscillatory architecture paces population activity for
offline reactivation across the human medial temporal lobe.* **Neuron 114,
3786-3798.** [DOI: 10.1016/j.neuron.2026.05.004](https://doi.org/10.1016/j.neuron.2026.05.004).

Read the **STAR Methods > Detection of hippocampal ripples**, especially
**"Loose detection of putative ripple events"** and
**"Template matching and white-matter control"** (page e9; PDF page 23 of the
25-page article supplied with the original analysis). The paper also explains
how templates were trained; **this repository deliberately does not train or
recalculate them**.

> **Scope / validation:** This is a generalized implementation of the supplied
> analysis scripts, not a numerically certified reproduction or a clinically
> validated detector. The included templates and signals are synthetic. Use
> experimentally validated templates and benchmark against annotated LFPs
> before making physiological claims.

## What is included

```text
swr-template-detector/
|-- general_swr_detection_with_loader.py   # Single standalone script: inputs, functions, example
|-- demo_ripple_templates.npz             # Synthetic templates for runnable demonstration ONLY
|-- README.md
|-- CITATION.md
|-- docs/
|   |-- METHODS_AND_IMPLEMENTATION.md     # Paper-to-implementation method and differences
|   |-- QUICK_CROSSWALK.md                # Short original-to-new guide
|   |-- ORIGINAL_CODE_MAPPING.md          # Detailed crosswalk to original laboratory scripts
|   `-- OUTPUT_SCHEMA.md                  # Dictionary keys and index conventions
`-- tests/test_detector.py               # Synthetic sanity tests
```

## Install and run the 20-minute demonstration

Python 3.10+ recommended.

```bash
git clone https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python general_swr_detection_with_loader.py
```

Before the first push, replace `YOUR-ACCOUNT/YOUR-REPOSITORY` with the GitHub
repository you created. To run the downloaded ZIP instead of cloning, unzip it,
change into `swr-template-detector/`, install requirements, and run the script.

By default, the script makes two **independent Gaussian-noise recordings**,
20 minutes long at 1,250 Hz. It loads the bundled synthetic `.npz` template
library and writes (ignored by Git) `demo_swr_candidates.csv` and
`demo_swr_results.pkl` in the same directory. A small number of accepted noise
patterns is possible: these are **false positives**, not biological SWRs.

### Main steps and paper mapping

| Stage | New functions | STAR Methods subsection |
| --- | --- | --- |
| Supply templates | `load_templates` | "Dimensional embedding and template generation" (background only; generation excluded here) |
| Band-limited amplitude | `bandPass`, `get_smooth` | "Loose detection of putative ripple events" |
| Adaptive loose candidates | `detect_burst_events`, `detect_burst_events_adaptive_thresh`, `exclude_bad_windows` | "Loose detection of putative ripple events" |
| Narrowband alignment | `get_wv_hilbert`, `get_peakClosest2Time` | "Loose detection of putative ripple events" |
| Match probe and control | `getPeaksWithinWindows`, `extract_snippets`, `match_cosine_or_pearson` | "Template matching and white-matter control" |
| Accept/reject | `categorize_event_matches`, `detect_swrs` | "Template matching and white-matter control" |
| Reopen saved results | `save_swr_results`, `load_swr_new` | Implementation-level convenience; analogous to original `bb.load_swr_new` |

The paper describes initial **sleep-based** ripple detection and template
creation, but these are *prerequisites* for this repository and are not run.

## Use your own LFPs and your existing templates

In the **USER INPUTS** section of
[`general_swr_detection_with_loader.py`](general_swr_detection_with_loader.py),
set:

```python
use_simulated_data = False
lfp = np.load("probe_lfp.npy")           # 1D probe recording
ref = np.load("white_matter_lfp.npy")   # 1D synchronized control
sr = 1250.0                             # Hz, both signals
templates_path = Path("validated_templates.pkl")
sr_templates = 1250.0                   # sample rate of templates
thresh = 0.85
```

Both signals must be **one-dimensional, finite, the same length, time-aligned,
and have the same sampling rate**. The `ref` recording must be a real
anatomically appropriate white-matter control if the result is to be
interpreted with the paper's spatial-specificity criterion. Preprocessing,
reference montage, amplitude scale, and template provenance affect the
comparison.

Template file formats:

- `.pkl`: trusted Python pickle containing a dict with a `templates` key, or a
  bare `(n_templates, n_timepoints)` array. This supports the original saved
  template format. **Do not open untrusted pickle files.**
- `.npz`: NumPy archive with `templates` key.
- `.npy`: bare 2D NumPy array.

Templates are **never retrained**; if their sample rate differs from `sr`, the
loader only *resamples the existing waveforms* and issues a warning. The
shape and snippet window must be consistent with your validated template
library (details in [Methods notes](docs/METHODS_AND_IMPLEMENTATION.md)).

For code that loads the LFPs from your own source, see
[`examples/use_own_lfp.py`](examples/use_own_lfp.py), which demonstrates
calling the detector without executing its built-in demo.

## User settings (edit at the top of the main script)

| Setting | Default | Role |
| --- | --- | --- |
| `sr`, `sr_templates` | 1250 Hz | Recording and template sampling rates |
| `templates_path` | synthetic `.npz` | Path to fixed templates |
| `use_simulated_data`, `duration_min`, `rand_seed` | `True`, 20, 42 | Demo controls |
| `rippleBand` | `[60, 180]` Hz | Band for loose event detection (paper) |
| `ripple_band` | `[50, 250]` Hz | Band for waveform matching in supplied clean-template script |
| `filter_order` | 4 | Butterworth filter order |
| `N`, `sigma` | 100, 50 at 1250 Hz | Gaussian smoothing window and SD; 80 ms and 40 ms |
| `smooth_envelope`, `smooth_divisor` | `True`, `"N"` | Whether/how to smooth envelope (original `/N` convention) |
| `envelope_normalization` | `"zscore"` | Normalize Hilbert envelope or leave raw; generalized option |
| `min_events_mn`, `max_events_mn` | 20, 30 | Target **loose candidate** rate (per minute), not accepted ripple rate |
| `min_separation_ms`, `min_duration_ms` | 150, 20 | Event constraints |
| `t_min`, `t_max`, `max_iter` | 0, 5, 50 | Adaptive threshold search bounds and iterations |
| `first_event_seen_s` | 0.5 | Edge-event rejection margin |
| `freq_realign`, `nCys_realign`, `winCheck_s_realign` | 80 Hz, 4, 0.05 s | Alignment-stage wavelet and peak search |
| `thresh` | 0.85 | Template-match cutoff applied in both contacts |
| `lag_samples` | 0 | Allowed lag in template matching (0 matches supplied main call) |
| `bad_windows` | `None` | Excluded half-open sample intervals `[start, end)` |
| `taLen_s` | 0.25 | **Total** broadband LFP snippet duration (not plus/minus 250 ms) |
| `save_csv`, `save_pkl`, `savefname`, `swr_results_path` | enabled | Output control and paths |

**Sampling-rate note:** The script scales `N` and `sigma` from their original
1,250 Hz sample units when `sr` changes. Ensure filtering bands remain below
Nyquist. The signal used for thresholding and its normalization should be
validated when changing datasets.

## Selection rule: probe-only template match

For each candidate, choose the maximum similarity over the supplied templates
separately for probe and white-matter control:

```python
keep = (best_scores >= thresh) & (best_scores_ref < thresh)
```

At `thresh = 0.85`:

| Probe match | White-matter match | Outcome |
| --- | --- | --- |
| `>= 0.85` | `< 0.85` | Accepted SWR |
| `< 0.85` | any | Probe nonmatch (`RETURN_BAD_INSTEAD=True`) |
| `>= 0.85` | `>= 0.85` | Excluded overlap (matches both contacts) |

`>=` matches the **original software's** boundary convention; the paper says
"exceeded 0.85" in prose. The published 0.85 cutoff was based on the 95th
percentile of white-matter scores in the paper's dataset. It is **not** a
universally calibrated threshold for arbitrary LFPs or other template sets.

## Read accepted events or reapply the threshold later

After running the main script, retrieve accepted SWRs:

```python
from general_swr_detection_with_loader import load_swr_new

swrDic = load_swr_new("demo_swr_results.pkl", thresh=0.85)
events = swrDic["events"]              # (n_accepted, 3): onset, peak, offset in samples
peaks = swrDic["peaks"]                # sample timestamps
best_scores = swrDic["best_scores"]    # corresponding probe match scores
```

To select the original code's "bad" events (probe nonmatches, **not**
probe-and-white-matter matches):

```python
swrDic_bad = load_swr_new(
    "demo_swr_results.pkl", thresh=0.85, RETURN_BAD_INSTEAD=True
)
```

Reapplying `thresh` does **not** recalculate events or matching scores. It
only updates which previously scored candidates are selected. For example:

```bash
python examples/load_and_rethreshold.py demo_swr_results.pkl
```

The saved `.pkl` stores the whole candidate set. The `.csv` stores one row per
matchable candidate with onset, peak, offset, both match scores, template
indices, and `accepted` according to the threshold at detection time. For
all keys and indexing details, see [Output schema](docs/OUTPUT_SCHEMA.md).

**Compatibility note:** This `load_swr_new(fname, ...)` consumes the *new
single-file result format*, not the original `bb.load_swr_new(seshLab, chreg)`
files. The old loader reconstructs paths and opens separate probe/control
pickles. The new version works without subject/session-specific conventions.
It does **not** compute `snipSpect`; that field is absent unless added by an
external analysis. This is intentional, not an omission of saved data.

## Test

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs these synthetic checks on Python 3.10-3.12.
They test loading/resampling, score semantics, threshold boundaries, excluded
windows, re-thresholded loading and a short end-to-end noise trace. **These
checks do not establish sensitivity, specificity, or scientific equivalence**
to the original analysis on experimental recordings.

## Reproducibility and limitations

1. **Validated templates required.** Synthetic demonstration templates must
   never be used as evidence of a real physiological ripple.
2. **White-matter control required for published classification.** Without a
   meaningful spatial control, `best_scores_ref < thresh` is not the paper's
   validation procedure.
3. **Generalization changes some implementation details.** Candidate-envelope
   normalization, numerical filtering, adaptive-search fallback, and nonzero
   lag matching are described explicitly in
   [Methods notes](docs/METHODS_AND_IMPLEMENTATION.md).
4. **Paper vs supplied scripts have parameter differences.** A 25-ms snippet
   is described during sleep template derivation, but the provided
   clean-template matching script uses a 25-ms **half-window**, yielding
   approximately 50-ms snippets. This repository derives snippet length from
   your loaded templates and does not silently force either number.
5. **No wavelet spectrogram output.** The standalone core returns broadband
   LFP snippets, not the downstream 10-200 Hz spectral analysis from the paper.
6. **Not a diagnostic device.** Clinical use is outside the scope of this code.
7. **Privacy:** Do not commit patient data, original results, or identifiable
   metadata. Binary results and common electrophysiology file types are in
   `.gitignore`, but review `git status` before every push.

## Detailed documentation

- [Methods and implementation notes](docs/METHODS_AND_IMPLEMENTATION.md)
- [Quick original-to-new code crosswalk](docs/QUICK_CROSSWALK.md)
- [Detailed old-to-new code mapping](docs/ORIGINAL_CODE_MAPPING.md)
- [Saved result schema](docs/OUTPUT_SCHEMA.md)
- [Scientific citation](CITATION.md)
- [Licensing before release](LICENSE-NOTICE.md)

## Before publishing on GitHub

Create your repository; choose and add a software `LICENSE` only if you are
authorized to license the code; replace the placeholder clone URL above;
verify your choice of real template distribution permissions; run the tests;
and inspect the repository for private data. Neither the original lab scripts
nor the article PDF nor any validated human templates are bundled.
