# sensors-noise

Code accompanying the article

> S. Moshfeghi and J. Jang, "Noise-Aware Learning for Traffic Safety Sensing: Propagation of
> Simulated Tracking Errors Through Surrogate Safety Measures and Denoising Models," *Sensors*, 2026.

The package injects simulated tracking errors into drone-recorded reference trajectories, reruns a
plausibility-filtered time-to-collision (TTC) and post-encroachment time (PET) conflict screen,
matches the resulting conflicts to the clean reference by road-user identity, and compares
classical and learned position cleaning (Savitzky–Golay, Kalman–RTS, two bidirectional GRU
denoisers and a gated hybrid), applied both to raw detections and to tracker-like output.

## Contents

| Path | Content |
|---|---|
| `sensors_noise/` | Python package: data loading, kinematics, conflict screen, error models, cleaning methods, evaluation, figures |
| `scripts/` | Command-line entry points, numbered in execution order |
| `notebooks/` | Notebooks that run or display each stage (distributed without outputs) |
| `tests/` | Unit tests against hand-computed values |
| `environment.yml` | Conda environment |

## Data (not included)

No data, results or trained models are redistributed in this repository. All results are
reproduced by the scripts from the public datasets below; please cite them as their authors request.

- **SinD** (signalized-intersection drone dataset): available from the dataset authors' public
  repository. Cite Xu et al., ITSC 2022, and Li et al., *SinD 2.0*, arXiv:2607.16943.
  The loader searches the data folder recursively for `Veh_smoothed_tracks.csv` and
  `Ped_smoothed_tracks.csv`, so the original city sub-folders can be kept.
- **COSMOS** trajectory release (New York City), Hugging Face `harrypotter1501/cosmos-trajectory`.
  Cite Zang et al. (2024) and Raychaudhuri et al. (MobiCom 2020). Used only for the
  roadside-deployment comparison (`scripts/04_cosmos_stats.py`).

## Installation

```bash
conda env create -f environment.yml
conda activate sensors-noise
# PyTorch is required for the learned denoisers and predictors; install the build that matches
# your platform and GPU driver from https://pytorch.org/get-started/locally/
python -m pytest tests -q
```

The experiments in the article were run with Python 3.11 and PyTorch 2.5.1.

## Reproducing the article

Run from the repository root, with SinD in `sind/` and COSMOS in `cosmos-trajectory/`
(or pass other paths). Experiments are seeded and resumable: finished tasks are cached under
`results/*/parts/` and skipped on rerun. `--jobs N` runs tasks in parallel (on Windows, use the
scripts rather than the notebooks for parallel runs).

| Article | Command |
|---|---|
| Sections 3.2–3.3, 4.1: conflict inventory and plausibility criteria | `python scripts/01_conflict_check.py --data-root sind` |
| Section 3.3: example events | `python scripts/02_plot_events.py --data-root sind` |
| Sections 3.4–3.5, 4.2: error propagation | `python scripts/03_noise_experiments.py --data-root sind --jobs 4` |
| Section 3.6: denoiser training | `python scripts/05_train_denoisers.py --data-root sind` |
| Section 3.6: tuning of classical methods and hybrid gate | `python scripts/05b_tune_classical.py --data-root sind` |
| Sections 3.6, 4.3: cleaning of raw detections, held-out errors | `python scripts/06_filter_experiments.py --data-root sind --jobs 4` |
| Sections 3.6, 4.3: cleaning applied to tracker output | `python scripts/06b_chain_experiments.py --data-root sind --jobs 4` |
| Sections 3.9, 4.4: roadside-deployment comparison | `python scripts/04_cosmos_stats.py --cosmos-root cosmos-trajectory --sind-root sind` |
| Sections 3.8, 4.2–4.3: analytical-model check and additional analyses | `python scripts/10_additional_analyses.py --results results` then `python scripts/10b_additional_analyses.py --results results` |
| Supplementary Note S1: prediction experiment | `python scripts/07_train_predictors.py --data-root sind` then `python scripts/08_predictor_experiments.py --data-root sind` |
| Figures | `python scripts/09_paper_figures.py --data-root sind --out paper_figures` |

`03_noise_experiments.py` and `06_filter_experiments.py` accept `--quick` (one seed, reduced grid)
for a short check and `--collect` to rebuild the tables from the cached parts.

## Method summary

### Kinematics
Velocity is derived from positions by central differences over ±0.5 s, and heading from
velocity above 0.5 m/s. The same derivation is applied to the clean reference and to every
noisy or cleaned condition, so each comparison isolates the effect of the errors. Detection is
also reported against a reference built from SinD's native kinematics.

### Conflict screen
- **TTC**: constant-velocity extrapolation in two stages, a multi-disc screen giving a lower
  bound and an exact recomputation with oriented rectangles (bisection to about 2 ms). Only pairs
  within 30 m that are closing at 1 m/s or more are evaluated; pedestrian–pedestrian pairs are
  skipped.
- **TTC conflict**: minimum TTC below the threshold for at least 3 frames, and an actual
  edge-to-edge footprint gap of at most 1 m within 3 s after the minimum-TTC moment.
- **PET**: computed at the native 10 Hz with sub-sample interpolation. The conflict point is the
  midpoint of the closest crossing sample pair; PET is the time from the first road user's
  reference point leaving a disc of radius 0.5 (w_a + w_b) around it to the second road user's
  entering it. Only crossing motions (30–150° between directions of motion) are considered.
- **Plausibility criteria**: a pair is removed if either road user is static (extent of the track
  below 3 m diagonal, after a 2-s rolling median and on the 2nd–98th percentile extent) or if the
  two footprints overlap for more than 0.3 s in total.
- **Grouping**: geometry class from the angle at minimum TTC (same direction < 30°, crossing
  30–150°, opposite > 150°); road-user groups vehicle–vehicle, VRU–vehicle and VRU–VRU
  (VRU = pedestrian or two-wheeler). Results are reported pooled and per recording.

### Error models (`sensors_noise/noise.py`)

| Kind | Levels | Model |
|---|---|---|
| `jitter` | 0.1, 0.3, 0.5, 1.0 m | Independent Gaussian per frame; level = mean 2-D error |
| `distance_jitter` | 0.3, 0.5 m at 40 m | Jitter scaled by distance to a virtual camera at the scene corner (floor 0.25×) |
| `drift` | 0.3, 0.5 m | AR(1) per road user and axis, 1-s correlation time |
| `offset` | 0.3, 0.6, 1.0 m | Constant shift along the heading per road user, random sign |
| `heading` | 2, 5, 10° | Gaussian heading error per frame; affects footprint orientation |
| `dropout` | 5, 15, 30 % | Missed detections in bursts with a mean length of 0.5 s |
| `id_switch` | 0.5, 1, 3 per VRU track | Poisson number of track splits (vehicles at one third of the rate), 0.3-s re-identification gap |

Held-out conditions, never used in denoiser training: `jitter_t` (Student-t jitter, 3 degrees
of freedom, 0.3 m), `drift_slow` (3-s correlation time, 0.5 m) and `dropout_long` (1.5-s bursts, 15 %).

### Processing conditions and cleaning methods

| Method | Description |
|---|---|
| `none` | Raw condition: kinematics derived directly from the noisy positions, no filtering |
| `tracker` | Tracker-like condition: causal constant-velocity Kalman filter with fixed measurement noise and gap filling by prediction |
| `savgol` | Gap filling, then Savitzky–Golay smoothing of positions |
| `kalman` | Constant-velocity Kalman filter with RTS smoother; measurement noise estimated per track from second differences |
| `denoiser_jitter` | BiGRU denoiser trained on Gaussian jitter only (0–1 m) |
| `denoiser_mixed` | Same architecture and data volume, trained on jitter, distance jitter, drift, offset and missed detections (matched distribution) |
| `hybrid` | Per-track noise estimate routes each track to no cleaning, the Kalman smoother or the matched-distribution denoiser |
| `tracker+<method>` | The cleaning method applied to tracker-like output (`06b_chain_experiments.py`) |

All methods clean positions and fill missed frames only; kinematics are then derived as for the
reference. Classical parameters and the hybrid gate are tuned on training-seed errors
(`05b_tune_classical.py`, written to `results/03_filters/models/tuning.json`), never on the
evaluation seeds. The denoisers use a two-fold split by road user, so each road user is cleaned by
the model trained on the other fold, and training errors use seeds disjoint from the evaluation seeds.

### Evaluation (`sensors_noise/noise_eval.py`)
Conflicts are matched to the clean reference by the original identities of both road users, so a
conflict split by an identity switch still counts as found. Reported per error kind, level,
recording and road-user group: recall, precision, missed and false conflicts, severity-shift rate,
and median, 90th-percentile and signed TTC/PET errors on conflicts found in both. Confidence
intervals for recall and precision are obtained by a cluster bootstrap over road-user pairs. Seeds
are derived from (recording, kind, level, seed) with a stable checksum, so results do not depend
on run order or the number of workers.

## Citation

If you use this code, please cite the article above (see also `CITATION.cff`).

## License

MIT; see `LICENSE`. The license covers this code only, not the SinD or COSMOS data.
