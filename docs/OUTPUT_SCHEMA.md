# Output dictionary schema and indexing conventions

`results = detect_swrs(lfp, ref, templates, sr, bad_windows=...)` returns a
dictionary. Timing fields use **sample indices in the shared LFP timebase**;
for example `results['peaks'] / sr` yields times in seconds.

## Event table convention

`events` and `events_candidates` have shape `(n_events, 3)`:

```text
column 0: onset index
column 1: ripple-band peak index in the probe
column 2: offset index
```

The burst onset/offset are from the **probe's loose candidate window**, while
its peak is **realigned to the local maximum of the probe's matching-band
signal**. Event windows have inclusive ends in this detector. Artifact input
intervals (`bad_windows`) are half-open `[start,end)`.

## Candidate-aligned fields

The following arrays are indexed in the same order and have one entry per
**matchable candidate** (after edge and bad-window exclusions):

| Key | Shape / meaning |
| --- | --- |
| `events_candidates` | `(n,3)` loose event onset, matching-band probe peak, offset |
| `peaks_candidates` | `(n,)` matching-band probe peaks |
| `peaks_ref_candidates` | `(n,)` matching-band WM peaks in *same* loose event windows |
| `troughs_candidates` | `(n,)` narrowband 80-Hz troughs, probe |
| `peaks_80hz_candidates` | `(n,)` narrowband 80-Hz peaks, probe |
| `snippets_candidates` | `(n,L)` filtered probe matching snippets; `L=templates.shape[1]` |
| `snippets_ref_candidates` | `(n,L)` filtered WM matching snippets |
| `snipLFPs_candidates` | `(n,T)` z-scored broadband probe snippets, total `T ~= taLen_s*sr` |
| `best_scores` | `(n,)` maximum probe template similarities |
| `best_scores_ref` | `(n,)` maximum WM template similarities |
| `best_tidx` | `(n,)` winning probe template indices (zero-based) |
| `best_tidx_ref` | `(n,)` winning WM template indices (zero-based) |
| `best_lag`, `best_lag_ref` | `(n,)` best lags in samples, usually zero |
| `best_full` | `(n,K)` probe scores across templates at winning lag, `K=n_templates` |
| `good_mask` | `(n,)` accepted boolean flags at the runtime `thresh` |

**Selected fields:**

- `events`: accepted candidates `(n_accepted,3)`.
- `peaks`: accepted probe matching-band peak indices.
- `idx_orig_only`: candidate indices accepted by probe-only rule.
- `idx_nonmatch`: candidate indices with probe score below threshold.
- `idx_orig_and_ref`: candidate indices matching in both probe and WM.

**Axes / run metadata:** `ta_axis` (broadband window axis in **milliseconds**),
`threshold_used` (adaptive candidate threshold), `n_loose` (number of loose
candidates **before edge exclusion**), `min_events`, `max_events` (per-trace
count targets), and `sr`.

**Not generated:** `snipSpect`, `freqs_spectro` and the full original plotting
or single-neuron analyses. Only report them if you compute them separately.

## Loaded selected dictionary

```python
swrDic = load_swr_new("results.pkl", thresh=0.85)
```

Returns selected `events`, `peaks`, `snippets`, `best_scores`,
`best_scores_ref`, available `snipLFPs`, `best_tidx`, `best_tidx_ref`,
`best_lag`, `best_lag_ref`, full candidate category indices, `thresh`, `sr`,
and `ta_axis` when present.

`RETURN_BAD_INSTEAD=True` returns `idx_nonmatch` candidates (probe score below
threshold), not events that matched in **both** signals. The category index
arrays in the returned dictionary refer to the *unfiltered candidate index*
space; selected arrays refer only to the subset requested.

`save_candidates_csv` writes candidate-level scalar metadata and one boolean
`accepted` value based on the threshold **used at CSV export time**. Loading
with a new threshold does not automatically update an existing CSV; export a
new one if a different classification is required.

## Stability and file safety

The pickle format is a Python-specific serialization intended for **trusted
local results** only. Never unpickle unknown external files. Retain your code
version and detector settings with published analyses, because changing
`thresh` is supported while changing filtering, alignment or template library
requires **rerunning** `detect_swrs`.
