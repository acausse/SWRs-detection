#!/usr/bin/env python3
"""Standalone, single-probe SWR detector with a white-matter control.

Only numpy and scipy are required for detection. Templates are LOADED, never
trained or generated here. Optional CSV output uses Python's standard library.

Pipeline: 60-180 Hz envelope -> adaptive candidate detection -> peak alignment
-> template matching in probe and white matter -> spatial specificity rule.

Source: Causse et al., Neuron (2026), STAR Methods, "Detection of hippocampal
ripples". In particular, the final acceptance rule is
    best_scores >= thresh AND best_scores_ref < thresh.

IMPORTANT: The demo template file is SYNTHETIC and is for testing this program
only. Substitute your experimentally validated templates for real detection.
"""

from pathlib import Path
from fractions import Fraction
import csv
import pickle
import warnings

import numpy as np
from scipy import signal as sig


# =============================================================================
# 1. USER INPUTS -- MODIFY THIS SECTION
# =============================================================================

# Supply an existing .pkl (dict containing 'templates'), .npz (key 'templates'),
# or .npy (2D template array). A template has shape (n_samples,).
templates_path = Path(__file__).with_name("demo_ripple_templates.npz")
sr_templates = 1250.0     # Hz; acquisition rate at which templates were saved

# Example data; replace with your real lfp and ref arrays below.
use_simulated_data = True
# For real recordings, set use_simulated_data=False and supply two 1D arrays:
# lfp = np.load("your_probe_lfp.npy")
# ref = np.load("your_white_matter_lfp.npy")
lfp = None
ref = None
sr = 1250.0               # Hz; sample rate of BOTH lfp and ref
duration_min = 20.0       # simulation only
rand_seed = 42           # simulation only

# Filter settings.
rippleBand = [60.0, 180.0]  # Hz; candidate detection (paper)
ripple_band = [50.0, 250.0]  # Hz; matching, as in your clean-template script
filter_order = 4

# Candidate envelope; the original N=100, sigma=50 at 1250 Hz means
# an 80 ms Gaussian window with sigma=40 ms, as described in the paper.
N = 100                   # Gaussian window width, in samples AT 1250 Hz
sigma = 50.0              # Gaussian SD, in samples AT 1250 Hz
smooth_envelope = True
# 'zscore' mimics the normalized HFB detection branch; 'raw' mimics the
# original rip branch before smoothing. For arbitrary recording units,
# 'zscore' is generally easier to threshold reproducibly.
envelope_normalization = "zscore"   # 'zscore' or 'raw'
# Preserve original ac.get_smooth(...)/N behavior; 'kernel_sum' uses unit area.
smooth_divisor = "N"     # 'N' or 'kernel_sum'

# Adaptive event detector, as in ac_run_detect_ffaBursts_macro.py.
min_events_mn = 20.0
max_events_mn = 30.0
min_separation_ms = 150.0
min_duration_ms = 20.0
t_min = 0.0
t_max = 5.0
max_iter = 50
first_event_seen_s = 0.5

# Alignment to the 80-Hz wavelet (also return troughs for comparison).
freq_realign = 80.0
nCys_realign = 4
winCheck_s_realign = 0.05

# Template classification.
thresh = 0.85
lag_samples = 0         # zero-lag Pearson-like cosine score, as in the script

# Optional excluded intervals: [onset_sample, offset_sample) pairs.
# For times in seconds use (np.asarray(windows_s) * sr).astype(int).
bad_windows = None

# Results: CSV includes every candidate, its two scores and accepted flag.
save_csv = True
savefname = Path(__file__).with_name("demo_swr_candidates.csv")

# Save ALL loose candidates and scores once; reload/filter with any thresh later.
# This is the generalized equivalent of bb.load_swr_new(..., thresh=thresh).
save_pkl = True
swr_results_path = Path(__file__).with_name("demo_swr_results.pkl")

# Broadband LFP snippets around the local ripple-band peaks; no full spectrogram
# is calculated by this minimal detector. The original paper uses 250 ms here.
taLen_s = 0.25


# =============================================================================
# 2. FUNCTIONS -- NO CUSTOM LAB MODULES REQUIRED
# =============================================================================

def load_templates(fname, sr_templates, sr):
    """Load existing templates; resample templates only if sample rates differ.

    The window length for matching is derived from the loaded templates, not
    hard-coded. Consequently, an original 62-sample template produces a
    62-sample candidate snippet (31 samples on either side of the peak).
    """
    fname = Path(fname)
    if not fname.is_file():
        raise FileNotFoundError(
            f"Template file not found: {fname}\n"
            "Set templates_path to your existing validated template file."
        )
    if fname.suffix.lower() == ".pkl":
        # IMPORTANT: use pickle only with files you trust.
        with fname.open("rb") as handle:
            data = pickle.load(handle)
        templates = data["templates"] if isinstance(data, dict) else data
    elif fname.suffix.lower() == ".npz":
        with np.load(fname, allow_pickle=False) as data:
            templates = data["templates"]
    elif fname.suffix.lower() == ".npy":
        templates = np.load(fname, allow_pickle=False)
    else:
        raise ValueError("Template file must be .pkl, .npz or .npy")

    templates = np.asarray(templates, dtype=np.float64)
    if templates.ndim != 2 or templates.shape[0] < 1 or templates.shape[1] < 8:
        raise ValueError("templates must be (n_templates, n_timepoints), >=8 samples")
    if not np.all(np.isfinite(templates)) or np.any(np.std(templates, axis=1) == 0):
        raise ValueError("Templates contain non-finite values or flat waveforms")
    if sr <= 0 or sr_templates <= 0:
        raise ValueError("sr and sr_templates must be positive")

    if not np.isclose(sr_templates, sr):
        # Resample only the fixed, short template waveforms; leave LFP untouched.
        ratio = Fraction(float(sr / sr_templates)).limit_denominator(10000)
        templates = sig.resample_poly(
            templates, ratio.numerator, ratio.denominator, axis=1
        )
        warnings.warn(
            f"Resampled templates from {sr_templates:g} Hz to {sr:g} Hz; "
            "verify waveform alignment before using on real data."
        )
    return templates


def bandPass(signal, low, high, sr, filtorder=4):
    """Zero-phase Butterworth bandpass, stable second-order sections."""
    if not (0 < low < high < sr / 2):
        raise ValueError(f"Invalid band [{low}, {high}] Hz for sr={sr} Hz")
    sos = sig.butter(filtorder, [low, high], btype="bandpass", fs=sr, output="sos")
    return sig.sosfiltfilt(sos, signal)


def get_smooth(signal, N, sigma, smooth_divisor="N"):
    """Gaussian smoothing with original script's even-N center convention."""
    kernel = sig.windows.gaussian(N, std=sigma)
    divisor = N if smooth_divisor == "N" else kernel.sum()
    if smooth_divisor not in ("N", "kernel_sum"):
        raise ValueError("smooth_divisor must be 'N' or 'kernel_sum'")
    # The original ac.get_smooth uses full convolution cropped from N//2.
    full = sig.fftconvolve(signal, kernel, mode="full")
    return full[N // 2:N // 2 + len(signal)] / divisor


def prepare_bad_windows(bad_windows, n_samples):
    """Clip and merge excluded half-open intervals, in sample indices."""
    if bad_windows is None:
        return np.empty((0, 2), dtype=np.int64)
    wins = np.asarray(bad_windows)
    if wins.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if wins.ndim != 2 or wins.shape[1] != 2 or not np.all(np.isfinite(wins)):
        raise ValueError("bad_windows must have shape (n,2) with finite values")
    wins = np.asarray(wins, dtype=np.int64).copy()
    wins[:, 0] = np.clip(wins[:, 0], 0, n_samples)
    wins[:, 1] = np.clip(wins[:, 1], 0, n_samples)
    wins = wins[wins[:, 1] > wins[:, 0]]
    wins = wins[np.argsort(wins[:, 0])]
    merged = []
    for start, end in wins:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([int(start), int(end)])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)


def detect_burst_events(zscore_signal, threshold, sr, min_separation_ms,
                        min_duration_ms):
    """Return [start, envelope peak, end] in samples, inclusive end.

    Mimics the original behavior: short events are dropped and, of two events
    closer than the separation criterion, the earlier accepted event is kept.
    """
    above = np.asarray(zscore_signal) > threshold
    transitions = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    min_dur = max(1, int(min_duration_ms * sr / 1000))
    min_sep = max(0, int(min_separation_ms * sr / 1000))
    valid = (starts > 0) & (ends < len(above) - 1) & (ends - starts + 1 >= min_dur)
    events = []
    for start, end in zip(starts[valid], ends[valid]):
        if events and start - events[-1][2] < min_sep:
            continue
        peak = int(start + np.argmax(zscore_signal[start:end + 1]))
        events.append([int(start), peak, int(end)])
    return np.asarray(events, dtype=np.int64).reshape(-1, 3)


def exclude_bad_windows(events, bad_windows):
    """Drop any candidate that intersects an excluded [start, end) interval."""
    if len(events) == 0 or len(bad_windows) == 0:
        return events
    keep = np.ones(len(events), dtype=bool)
    for start, end in bad_windows:
        keep &= ~((events[:, 0] < end) & (events[:, 2] >= start))
    return events[keep]


def detect_burst_events_adaptive_thresh(band_sm, bad_windows, sr,
                                        min_events, max_events,
                                        min_separation_ms, min_duration_ms,
                                        t_min, t_max, max_iter):
    """Binary-search a threshold targeting a number of artifact-free bursts.

    If no tested threshold achieves the count interval, return the tested
    threshold whose count is closest to the target, rather than pretending
    the target was achieved.
    """
    if not (0 <= min_events <= max_events and t_min < t_max and max_iter > 0):
        raise ValueError("Invalid event-count or adaptive-threshold settings")
    target = 0.5 * (min_events + max_events)
    best = None
    for _ in range(max_iter):
        threshold = 0.5 * (t_min + t_max)
        events = detect_burst_events(
            band_sm, threshold, sr, min_separation_ms, min_duration_ms
        )
        events = exclude_bad_windows(events, bad_windows)
        n_valid = len(events)
        if best is None or abs(n_valid - target) < best[0]:
            best = (abs(n_valid - target), events, threshold)
        if min_events <= n_valid <= max_events:
            return events, threshold
        if n_valid > max_events:
            t_min = threshold  # raise the threshold -> fewer events
        else:
            t_max = threshold  # lower the threshold -> more events
        if np.isclose(t_min, t_max, rtol=0, atol=1e-10):
            break
    return best[1], best[2]


def get_wv_hilbert(signal, sr, freq_realign, nCys_realign):
    """Single-frequency Morlet convolution analogous to ac.WVfiltering.

    Uses the original wavelet-length rule (10*sr/freq) and wavelet parameter
    nCys_realign. The wavelet is an alignment aid; matching ultimately uses
    the local peak of the matching-band signal, as in the supplied code.
    """
    n = max(9, int(10 * sr / freq_realign))
    x = np.linspace(-2 * np.pi, 2 * np.pi, n)
    w = float(nCys_realign)
    wave = np.pi**(-0.25) * np.exp(1j * w * x) * (
        np.exp(-0.5 * x**2) - np.exp(-0.5 * w**2)
    )
    envelope_cdf = np.cumsum(np.abs(wave)) / np.sum(np.abs(wave))
    right = np.flatnonzero(envelope_cdf[::-1] <= 1e-4)
    if len(right):
        cut = int(right[0])
        if 2 * cut < n - 3:
            wave = wave[cut:n - cut]
    wave = wave / (0.5 * np.sum(np.abs(wave)))
    return sig.fftconvolve(signal, wave, mode="same")


def get_peakClosest2Time(signal, tPts_sample, winCheck_s, sr, valley=False):
    """Find the closest local peak/trough in a centered temporal window."""
    half = max(1, int(round(winCheck_s * sr / 2)))
    peaks = np.empty(len(tPts_sample), dtype=np.int64)
    for i, t in enumerate(tPts_sample):
        start = max(0, int(t) - half)
        end = min(len(signal), int(t) + half + 1)
        chunk = signal[start:end]
        if valley:
            chunk = -chunk
        candidates, _ = sig.find_peaks(chunk)
        if len(candidates):
            local = candidates[np.argmin(np.abs((start + candidates) - t))]
        else:
            local = int(np.argmax(chunk))   # robust fallback for flat windows
        peaks[i] = start + local
    return peaks


def getPeaksWithinWindows(signal, event_intervals):
    """Peak of filtered signal within each [start, end] event interval."""
    return np.asarray([
        int(start + np.argmax(signal[start:end + 1]))
        for start, end in event_intervals
    ], dtype=np.int64)


def extract_snippets(signal, peaks, n_samples):
    """Fixed-length snippets centered on peaks; return validity mask."""
    left = n_samples // 2
    right = n_samples - left
    valid = (peaks >= left) & (peaks + right <= len(signal))
    offsets = np.arange(-left, right, dtype=np.int64)
    snippets = signal[peaks[valid, None] + offsets[None, :]]
    return snippets, valid


def match_cosine_or_pearson(templates, snippets, lag_samples=0):
    """Max centered cosine (Pearson r) per event, over templates and lags.

    Lag 0 matches acsw.match_cosine_or_pearson (to numerical precision).
    Unlike the original nonzero-lag implementation, each overlap is centered
    and normalized independently; no wraparound and no asymmetric norm bias.
    """
    templates = np.asarray(templates, dtype=np.float64)
    snippets = np.asarray(snippets, dtype=np.float64)
    n_events = len(snippets)
    n_templates, n_timepoints = templates.shape
    if snippets.shape != (n_events, n_timepoints):
        raise ValueError("Snippet length must match the loaded template length")
    if not (0 <= lag_samples < n_timepoints // 4):
        raise ValueError("lag_samples must be >=0 and < one quarter of window")
    best_scores = np.full(n_events, -np.inf)
    best_tidx = np.full(n_events, -1, dtype=int)
    best_lag = np.zeros(n_events, dtype=int)
    best_full = np.full((n_events, n_templates), np.nan)
    for lag in range(-lag_samples, lag_samples + 1):
        if lag >= 0:
            X, T = snippets[:, lag:], templates[:, :n_timepoints - lag]
        else:
            X, T = snippets[:, :n_timepoints + lag], templates[:, -lag:]
        X = X - X.mean(axis=1, keepdims=True)
        T = T - T.mean(axis=1, keepdims=True)
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        tn = np.linalg.norm(T, axis=1, keepdims=True)
        scores = (X @ T.T) / np.maximum(xn * tn.T, 1e-12)
        idx = np.argmax(scores, axis=1)
        rows = np.arange(n_events)
        score_here = scores[rows, idx]
        better = score_here > best_scores
        best_scores[better] = score_here[better]
        best_tidx[better] = idx[better]
        best_lag[better] = lag
        best_full[better] = scores[better]
    return best_scores, best_tidx, best_lag, best_full


def detect_swrs(lfp, ref, templates, sr, bad_windows=None):
    """Detect SWRs in ONE probe and ONE white-matter control LFP.

    Inputs:
        lfp, ref: one-dimensional arrays of equal length (same timebase).
        templates: fixed (n_templates, n_samples) library already loaded.
        bad_windows: excluded [onset, offset) intervals in SAMPLE indices.
    Outputs:
        results['events']: accepted [onset, peak, offset] sample indices.
        results['events_candidates']: every matchable candidate, in samples.
        'best_scores', 'best_scores_ref', 'best_tidx', 'best_tidx_ref', etc.
    """
    lfp = np.asarray(lfp, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if lfp.ndim != 1 or ref.ndim != 1 or len(lfp) != len(ref):
        raise ValueError("lfp and ref must be equal-length 1D arrays")
    if len(lfp) < sr * 2 or not np.all(np.isfinite(lfp)) or not np.all(np.isfinite(ref)):
        raise ValueError("Signals must be finite and at least 2 seconds long")
    if not (0 < rippleBand[0] < rippleBand[1] < sr/2):
        raise ValueError("rippleBand invalid for sr")

    n_samples = len(lfp)
    bad_windows = prepare_bad_windows(bad_windows, n_samples)
    min_events = int(n_samples / sr / 60 * min_events_mn)
    max_events = int(n_samples / sr / 60 * max_events_mn)

    # A. Envelope and loose adaptive detection, using the PROBE only.
    ripsig = bandPass(lfp, *rippleBand, sr=sr, filtorder=filter_order)
    ripsig_pow = np.abs(sig.hilbert(ripsig))
    if envelope_normalization == "zscore":
        std = ripsig_pow.std()
        if std <= 1e-12:
            raise ValueError("Ripple envelope is constant")
        band_sm = (ripsig_pow - ripsig_pow.mean()) / std
    elif envelope_normalization == "raw":
        band_sm = ripsig_pow
    else:
        raise ValueError("envelope_normalization must be 'zscore' or 'raw'")
    if smooth_envelope:
        # Define smoothing in ms implicitly using original 1250 Hz N/sigma.
        n_local = max(3, int(round(N * sr / 1250.0)))
        sd_local = max(1e-6, sigma * sr / 1250.0)
        band_sm = get_smooth(band_sm, n_local, sd_local, smooth_divisor)

    events, threshold_used = detect_burst_events_adaptive_thresh(
        band_sm, bad_windows, sr=sr,
        min_events=min_events, max_events=max_events,
        min_separation_ms=min_separation_ms, min_duration_ms=min_duration_ms,
        t_min=t_min, t_max=t_max, max_iter=max_iter
    )
    n_loose = len(events)

    # Remove edge events BEFORE wavelet alignment or snippet extraction.
    edge = int(round(first_event_seen_s * sr))
    left = templates.shape[1] // 2
    right = templates.shape[1] - left
    margin = max(edge, left, right, int(sr * winCheck_s_realign))
    events = events[(events[:, 0] >= margin) &
                    (events[:, 2] < n_samples - margin)]

    # B. Narrowband realignment as in the initial detector.
    if len(events):
        fSig = get_wv_hilbert(lfp, sr, freq_realign, nCys_realign)
        troughs = get_peakClosest2Time(np.real(fSig), events[:, 1],
                                      winCheck_s_realign, sr, valley=True)
        peaks_80hz = get_peakClosest2Time(np.real(fSig), events[:, 1],
                                        winCheck_s_realign, sr, valley=False)
    else:
        troughs = np.empty(0, dtype=np.int64)
        peaks_80hz = np.empty(0, dtype=np.int64)

    # C. As in ac_run_clean_swrs_templates.py, independently find LOCAL
    #    matching-band peaks in probe and WM, within the SAME candidate windows.
    ripple_band_lfp = bandPass(lfp, *ripple_band, sr=sr, filtorder=filter_order)
    ripple_band_ref = bandPass(ref, *ripple_band, sr=sr, filtorder=filter_order)
    event_intervals = events[:, [0, 2]]
    peaks = getPeaksWithinWindows(ripple_band_lfp, event_intervals)
    peaks_ref = getPeaksWithinWindows(ripple_band_ref, event_intervals)
    snippets, valid_lfp = extract_snippets(ripple_band_lfp, peaks, templates.shape[1])
    snippets_ref, valid_ref = extract_snippets(ripple_band_ref, peaks_ref, templates.shape[1])
    valid = valid_lfp & valid_ref
    if not np.all(valid):
        # Defensive handling; all corresponding arrays remain aligned.
        events = events[valid]
        peaks, peaks_ref = peaks[valid], peaks_ref[valid]
        troughs, peaks_80hz = troughs[valid], peaks_80hz[valid]
        snippets, _ = extract_snippets(ripple_band_lfp, peaks, templates.shape[1])
        snippets_ref, _ = extract_snippets(ripple_band_ref, peaks_ref, templates.shape[1])

    # D. Template scores and mandatory white-matter specificity.
    best_scores, best_tidx, best_lag, best_full = match_cosine_or_pearson(
        templates, snippets, lag_samples=lag_samples
    )
    best_scores_ref, best_tidx_ref, best_lag_ref, _ = match_cosine_or_pearson(
        templates, snippets_ref, lag_samples=lag_samples
    )
    good_mask = (best_scores >= thresh) & (best_scores_ref < thresh)
    idx_orig_only = np.flatnonzero(good_mask)
    idx_nonmatch = np.flatnonzero(best_scores < thresh)
    idx_orig_and_ref = np.flatnonzero((best_scores >= thresh) &
                                    (best_scores_ref >= thresh))

    events_candidates = events.copy()
    events_candidates[:, 1] = peaks

    # Broadband event-centered snippets for downstream LFP averages.
    # Keep them aligned with ALL candidates, not just accepted events.
    taLen = max(2, int(round(taLen_s * sr)))
    lfp_sd = float(lfp.std())
    if not np.isfinite(lfp_sd) or lfp_sd == 0:
        raise ValueError("Cannot z-score a flat LFP trace")
    lfp_z = (lfp - lfp.mean()) / lfp_sd
    snipLFPs, valid_lfp_broad = extract_snippets(lfp_z, peaks, taLen)
    if not np.all(valid_lfp_broad):
        raise ValueError("A candidate is too close to the trace edge for taLen_s")
    ta_axis = (np.arange(taLen) - taLen // 2) / sr * 1000.0  # ms

    return {
        "events": events_candidates[good_mask],
        "events_candidates": events_candidates,
        "peaks_candidates": peaks,
        "snippets_candidates": snippets,
        "snippets_ref_candidates": snippets_ref,
        "snipLFPs_candidates": snipLFPs,
        "ta_axis": ta_axis,
        "peaks": peaks[good_mask],
        "peaks_ref_candidates": peaks_ref,
        "troughs_candidates": troughs,
        "peaks_80hz_candidates": peaks_80hz,
        "best_scores": best_scores,
        "best_scores_ref": best_scores_ref,
        "best_tidx": best_tidx,
        "best_tidx_ref": best_tidx_ref,
        "best_lag": best_lag,
        "best_lag_ref": best_lag_ref,
        "best_full": best_full,
        "good_mask": good_mask,
        "idx_orig_only": idx_orig_only,
        "idx_nonmatch": idx_nonmatch,
        "idx_orig_and_ref": idx_orig_and_ref,
        "threshold_used": float(threshold_used),
        "n_loose": n_loose,
        "min_events": min_events,
        "max_events": max_events,
        "sr": sr,
    }


def save_candidates_csv(results, fname):
    """Save candidate timestamps, scores and final classifications."""
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    events = results["events_candidates"]
    with fname.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["onset_sample", "peak_sample", "offset_sample",
                         "peak_s", "best_scores", "best_scores_ref",
                         "best_tidx", "best_tidx_ref", "accepted"])
        for i, (onset, peak, offset) in enumerate(events):
            writer.writerow([int(onset), int(peak), int(offset),
                             float(peak) / results["sr"],
                             float(results["best_scores"][i]),
                             float(results["best_scores_ref"][i]),
                             int(results["best_tidx"][i]),
                             int(results["best_tidx_ref"][i]),
                             bool(results["good_mask"][i])])


def categorize_event_matches(best_scores, best_scores_ref, thresh=0.85):
    """Same three categories as acsw.categorize_event_matches.

    idx_nonmatch includes ANY probe nonmatch, regardless of the WM score.
    """
    best_scores = np.asarray(best_scores)
    best_scores_ref = np.asarray(best_scores_ref)
    if best_scores.shape != best_scores_ref.shape or best_scores.ndim != 1:
        raise ValueError("Probe and white-matter scores must be equal-length 1D arrays")
    if not -1 <= thresh <= 1:
        raise ValueError("thresh must be in [-1, 1] for cosine similarity")
    match_orig = best_scores >= thresh
    match_ref = best_scores_ref >= thresh
    idx_orig_only = np.flatnonzero(match_orig & ~match_ref)
    idx_nonmatch = np.flatnonzero(~match_orig)
    idx_orig_and_ref = np.flatnonzero(match_orig & match_ref)
    return idx_orig_only, idx_nonmatch, idx_orig_and_ref


def save_swr_results(results, fname):
    """Persist all candidate scores, timestamps, and snippets for later loading.

    Save *unfiltered* candidates, so `load_swr_new` can apply a new `thresh`
    without repeating filtering, event detection, or template matching.
    """
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    with fname.open('wb') as handle:
        pickle.dump(results, handle, protocol=4)


def load_swr_new(fname, thresh=0.85, RETURN_BAD_INSTEAD=False):
    """Standalone equivalent of bb.load_swr_new(seshLab, chreg, thresh).

    This version reads one detector result pickle (no lab-specific bblocks,
    session labels, or directory rules). It reproduces the exact categories in
    ac_baseObjects.bblocks.load_swr_new:

       good: best_scores >= thresh AND best_scores_ref < thresh
       bad:  best_scores < thresh    [only when RETURN_BAD_INSTEAD=True]
       both: best_scores >= thresh AND best_scores_ref >= thresh  [excluded]

    NOTE: As in the original, RETURN_BAD_INSTEAD=True does not return the
    both-contact events; use 'idx_orig_and_ref' to inspect these separately.

    Returns a dictionary with keys matching the original where available:
    'events', 'peaks', 'snippets', 'snipLFPs', 'best_scores', 'ta_axis'.
    'snipSpect' is not created by this detector and is intentionally absent.
    The reference scores and other per-event metadata are included too.
    """
    # Only open pickles produced by a trusted source.
    with Path(fname).open('rb') as handle:
        dic = pickle.load(handle)
    required = ('events_candidates', 'peaks_candidates', 'snippets_candidates',
                'best_scores', 'best_scores_ref')
    missing = [key for key in required if key not in dic]
    if missing:
        raise KeyError(f'Missing candidate fields in saved results: {missing}')

    n_events = len(dic['best_scores'])
    if any(len(dic[key]) != n_events for key in required):
        raise ValueError('Saved candidate arrays are not aligned')

    idx_orig_only, idx_nonmatch, idx_orig_and_ref = categorize_event_matches(
        dic['best_scores'], dic['best_scores_ref'], thresh=thresh
    )
    idx = idx_nonmatch if RETURN_BAD_INSTEAD else idx_orig_only
    outdic = {
        'events': np.asarray(dic['events_candidates'])[idx],
        'peaks': np.asarray(dic['peaks_candidates'])[idx],
        'snippets': np.asarray(dic['snippets_candidates'])[idx],
        'best_scores': np.asarray(dic['best_scores'])[idx],
        'best_scores_ref': np.asarray(dic['best_scores_ref'])[idx],
        'idx_orig_only': idx_orig_only,
        'idx_nonmatch': idx_nonmatch,
        'idx_orig_and_ref': idx_orig_and_ref,
        'thresh': thresh,
        'sr': dic['sr'],
    }
    # Optional metadata is subset only if it varies with each event.
    event_keys = {
        'snipLFPs': 'snipLFPs_candidates',
        'snipSpect': 'snipSpect_candidates',
        'best_tidx': 'best_tidx',
        'best_tidx_ref': 'best_tidx_ref',
        'best_lag': 'best_lag',
        'best_lag_ref': 'best_lag_ref',
    }
    for output_key, source_key in event_keys.items():
        if source_key in dic:
            arr = np.asarray(dic[source_key])
            if len(arr) != n_events:
                raise ValueError(f'Saved candidate field {source_key} is misaligned')
            outdic[output_key] = arr[idx]
    for key in ('ta_axis', 'freqs_spectro'):
        if key in dic:
            outdic[key] = dic[key]
    return outdic


# =============================================================================
# 3. EXAMPLE IMPLEMENTATION -- ONE 20-MIN TRACE + ONE WHITE-MATTER TRACE
# =============================================================================

if __name__ == "__main__":
    # By default, simulate two independent traces of broadband Gaussian noise.
    # To analyze real data, define lfp and ref in USER INPUTS above and set
    # use_simulated_data=False. The signals must share sr and sample timestamps.
    if use_simulated_data:
        rng = np.random.default_rng(rand_seed)
        n_samples = int(round(duration_min * 60 * sr))
        lfp = rng.standard_normal(n_samples)   # ONLY random noise: not real SWRs
        ref = rng.standard_normal(n_samples)   # independent white-matter noise
    elif lfp is None or ref is None:
        raise ValueError("Set lfp and ref in USER INPUTS when use_simulated_data=False")

    templates = load_templates(templates_path, sr_templates, sr)
    results = detect_swrs(lfp, ref, templates, sr, bad_windows=bad_windows)

    print(f"Recording: {len(lfp)/sr/60:.1f} min; sr={sr:g} Hz")
    print(f"Templates loaded: {templates.shape[0]} x {templates.shape[1]} samples")
    print(f"Adaptive threshold: {results['threshold_used']:.4f}")
    print(f"Target loose candidates: {results['min_events']} to {results['max_events']}")
    print(f"Loose candidates (before edge exclusion): {results['n_loose']}")
    print(f"Matchable candidates: {len(results['events_candidates'])}")
    print(f"Probe matches >= {thresh}: {np.sum(results['best_scores'] >= thresh)}")
    print(f"Probe + white-matter matches >= {thresh}: {len(results['idx_orig_and_ref'])}")
    print(f"Final accepted SWRs: {len(results['events'])}")
    if save_csv:
        save_candidates_csv(results, savefname)
        print(f"Saved candidate table: {savefname}")
    if save_pkl:
        save_swr_results(results, swr_results_path)
        # Reload with the same selection logic as your bb.load_swr_new method.
        swrDic = load_swr_new(swr_results_path, thresh=thresh)
        swrDic_bad = load_swr_new(swr_results_path, thresh=thresh,
                                  RETURN_BAD_INSTEAD=True)
        print(f"Loaded clean SWRs (HPC >= {thresh}, WM < {thresh}): "
              f"{len(swrDic['events'])}")
        print(f"Loaded probe nonmatches (RETURN_BAD_INSTEAD=True): "
              f"{len(swrDic_bad['events'])}")
        print(f"Loaded HPC+WM matches excluded: "
              f"{len(swrDic['idx_orig_and_ref'])}")
        print(f"Saved complete candidate results: {swr_results_path}")
    if not (results['min_events'] <= results['n_loose'] <= results['max_events']):
        print("NOTE: The adaptive search could not reach its target count on this trace.")
    print("NOTE: Noise and synthetic demonstration templates do not validate real SWR detection.")
