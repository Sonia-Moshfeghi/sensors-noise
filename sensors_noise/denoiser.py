"""Learned trajectory denoiser (step 3).

A small bidirectional GRU cleans one agent's track at a time, in overlapping 3-s windows
(30 frames at 10 Hz). Input per frame: noisy position relative to the window's mean
observed position, an observed/missed flag, and the road-user category. Output: a
correction added to the (gap-filled) noisy position. Missed frames are imputed.

Loss: Huber on positions plus Huber on frame-to-frame displacements (velocity), since
step 2 showed velocity errors drive TTC failures.

Two training variants, same architecture and data volume:
  'jitter'  - trained on independent Gaussian jitter only (the usual assumption)
  'mixed'   - noise-aware: trained on the full mix of position and detection noise
Evaluation uses a two-fold split by agent (crc32 of the original uid): each agent is
cleaned by the model trained on the other fold, and training noise uses seeds that are
disjoint from the evaluation seeds.
"""
from __future__ import annotations

import os
# Windows/conda: PyTorch and MKL each ship an OpenMP runtime; loading both aborts the
# process ("OMP: Error #15"). Allowing the duplicate is the standard workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:                       # load PyTorch's DLLs before NumPy/SciPy (Windows: WinError 127 otherwise)
    import torch  # noqa: F401
except ImportError:
    pass

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .filters import _finish, fill_gaps
from .noise import apply_noise

CATS = ("vehicle", "two_wheeler", "pedestrian")
WIN, STRIDE, SCALE = 30, 10, 10.0        # frames, frames, metres per unit
TRAIN_SEED_BASE = 1000                    # evaluation uses seeds 0, 1, 2

# Both variants include noise-free copies (jitter 0.0) so the model learns to leave
# accurate positions alone.
VARIANTS = {
    "jitter": {"jitter": [0.0, 0.1, 0.3, 0.5, 1.0]},
    "mixed": {"jitter": [0.0, 0.1, 0.3, 0.5, 1.0], "distance_jitter": [0.3, 0.5],
              "drift": [0.3, 0.5], "offset": [0.3, 0.6, 1.0],
              "dropout": [0.05, 0.15, 0.30]},
}


def agent_fold(uid: str) -> int:
    return zlib.crc32(str(uid).encode()) % 2


def _torch():
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() // 2)))   # workers share the CPU
    return torch


def build_model(hidden: int = 64, layers: int = 2):
    torch = _torch()
    nn = torch.nn

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(2 + 1 + len(CATS), hidden)
            self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * hidden, 2)
            nn.init.zeros_(self.out.weight)     # start as the identity: no correction
            nn.init.zeros_(self.out.bias)

        def forward(self, feats):
            h = torch.relu(self.inp(feats))
            h, _ = self.gru(h)
            return self.out(h)

    return Denoiser()


# --------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------

def make_windows(grid: pd.DataFrame, clean_xy: np.ndarray | None = None):
    """Cut gap-filled tracks into overlapping windows.

    Returns feats (N, WIN, 6), origin (N, 2), rows (N, WIN) row indices into `grid`
    (-1 = padding), fold (N,), and targets (N, WIN, 2) if clean_xy (aligned with grid
    rows) is given. Coordinates are relative to the origin and divided by SCALE.
    """
    xy = grid[["x", "y"]].to_numpy(float)
    obs = grid["observed"].to_numpy(bool)
    cat_idx = grid["category"].map({c: i for i, c in enumerate(CATS)}).fillna(0).astype(int).to_numpy()
    folds = grid["orig_uid"].map(agent_fold).to_numpy()
    rows_all = []
    for idx in grid.groupby("uid", sort=False).indices.values():
        n = len(idx)
        starts = list(range(0, max(n - WIN, 0) + 1, STRIDE))
        if n > WIN and starts[-1] != n - WIN:
            starts.append(n - WIN)
        for s in starts:
            r = np.full(WIN, -1)
            seg = idx[s:s + WIN]
            r[:len(seg)] = seg
            rows_all.append(r)
    if not rows_all:
        empty = np.zeros((0, WIN, 2 + 1 + len(CATS)), np.float32)
        return empty, np.zeros((0, 2)), np.zeros((0, WIN), int), np.zeros(0, int), None
    rows = np.stack(rows_all)
    valid = rows >= 0
    safe = np.where(valid, rows, rows[:, :1])         # pad with the first row's values
    o = obs[safe] & valid
    w = o.sum(1, keepdims=True)
    pos = xy[safe]
    origin = np.where(w > 0, (pos * o[..., None]).sum(1) / np.maximum(w, 1),
                      pos[:, 0])
    rel = (pos - origin[:, None, :]) / SCALE
    rel[~valid] = 0.0
    onehot = np.eye(len(CATS))[cat_idx[safe]]
    feats = np.concatenate([rel, o[..., None].astype(float), onehot], axis=-1).astype(np.float32)
    feats[~valid] = 0.0                     # padding frames are all-zero (incl. category)
    targets = None
    if clean_xy is not None:
        targets = ((clean_xy[safe] - origin[:, None, :]) / SCALE).astype(np.float32)
        targets[~valid] = 0.0
    return feats, origin, rows, folds[safe[:, 0]], targets


def build_training_set(recordings, variant: str, total_windows: int = 300_000,
                       seeds_per_condition: int = 1, verbose: bool = True):
    """Noisy/clean window pairs from clean recordings (list of trajectory tables).

    Each (recording, noise kind, level, seed) copy contributes an equal share of
    `total_windows`, so both variants see the same amount of data.
    """
    conds = [(k, lv) for k, levels in VARIANTS[variant].items() for lv in levels]
    copies = [(ri, k, lv, s) for ri in range(len(recordings)) for k, lv in conds
              for s in range(seeds_per_condition)]
    if variant == "jitter":     # match the mixed variant's number of copies
        n_mixed = len(recordings) * sum(len(v) for v in VARIANTS["mixed"].values()) * seeds_per_condition
        reps = int(np.ceil(n_mixed / len(copies)))
        copies = [(ri, k, lv, s + i * seeds_per_condition) for i in range(reps)
                  for (ri, k, lv, s) in copies][:n_mixed]
    per_copy = int(np.ceil(total_windows / len(copies)))
    rng = np.random.default_rng(TRAIN_SEED_BASE)
    F, T, Fo = [], [], []
    for ci, (ri, kind, level, s) in enumerate(copies):
        clean = recordings[ri]
        seed = TRAIN_SEED_BASE + 7919 * ci + s
        noisy = apply_noise(clean, kind, level, np.random.default_rng(seed))
        grid = fill_gaps(noisy)
        cxy = clean.set_index(["uid", "tkey"])[["x", "y"]]
        cxy = cxy.reindex(pd.MultiIndex.from_arrays([grid["orig_uid"], grid["tkey"]]))
        ok = cxy["x"].notna().to_numpy()
        grid, cxy = grid[ok].reset_index(drop=True), cxy[ok].to_numpy(float)
        feats, _, rows, folds, targets = make_windows(grid, cxy)
        enough = (feats[..., 2] > 0).sum(1) >= 5
        sel = np.flatnonzero(enough)
        if len(sel) > per_copy:
            sel = rng.choice(sel, per_copy, replace=False)
        F.append(feats[sel]); T.append(targets[sel]); Fo.append(folds[sel])
        if verbose:
            print(f"  [{ci + 1}/{len(copies)}] rec {ri} {kind} {level:g}: {len(sel)} windows")
    return np.concatenate(F), np.concatenate(T), np.concatenate(Fo)


def train_model(feats, targets, epochs: int = 10, batch: int = 256, lr: float = 1e-3,
                vel_weight: float = 1.0, device: str = "auto", verbose: bool = True, seed: int = 0):
    torch = _torch()
    torch.manual_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model = build_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    X = torch.from_numpy(feats)
    Y = torch.from_numpy(targets)
    valid = torch.from_numpy(np.abs(feats[..., 3:]).sum(-1) > 0)   # real frames have a category
    huber = torch.nn.SmoothL1Loss(reduction="none", beta=0.05)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            x, y, v = X[b].to(dev), Y[b].to(dev), valid[b].to(dev)
            pred = x[..., :2] + model(x)
            lp = (huber(pred, y).sum(-1) * v).sum() / v.sum().clamp(min=1)
            vv = v[:, 1:] & v[:, :-1]
            lv = (huber(pred[:, 1:] - pred[:, :-1], y[:, 1:] - y[:, :-1]).sum(-1) * vv).sum() \
                / vv.sum().clamp(min=1)
            loss = lp + vel_weight * 10.0 * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if verbose:
            print(f"    epoch {ep + 1}/{epochs}: loss {tot / n:.5f}")
    return model.cpu()


def train_variant(recordings, variant: str, out_dir, total_windows: int = 300_000,
                  epochs: int = 10, device: str = "auto", verbose: bool = True):
    """Train the two fold models of one variant and save them as <variant>_fold<k>.pt.
    Model k is trained on agents NOT in fold k and is later applied to fold k."""
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Building training windows for variant '{variant}'")
    feats, targets, folds = build_training_set(recordings, variant, total_windows, verbose=verbose)
    for k in (0, 1):
        sel = folds != k
        if verbose:
            print(f"  training {variant} fold {k} on {sel.sum()} windows")
        model = train_model(feats[sel], targets[sel], epochs=epochs, device=device,
                            verbose=verbose, seed=k)
        torch.save({"state_dict": model.state_dict(), "variant": variant, "fold": k},
                   out / f"{variant}_fold{k}.pt")
    return out


_MODEL_CACHE = {}


def load_models(models_dir, variant: str):
    torch = _torch()
    key = (str(models_dir), variant)
    if key not in _MODEL_CACHE:
        ms = []
        for k in (0, 1):
            ck = torch.load(Path(models_dir) / f"{variant}_fold{k}.pt", map_location="cpu",
                            weights_only=False)
            m = build_model()
            m.load_state_dict(ck["state_dict"])
            m.eval()
            ms.append(m)
        _MODEL_CACHE[key] = ms
    return _MODEL_CACHE[key]


def denoise(df: pd.DataFrame, models, batch: int = 4096) -> pd.DataFrame:
    """Clean a noisy trajectory table with the fold models; fills missed frames."""
    torch = _torch()
    grid = fill_gaps(df)
    feats, origin, rows, folds, _ = make_windows(grid)
    acc = np.zeros((len(grid), 2))
    cnt = np.zeros(len(grid))
    with torch.no_grad():
        for k in (0, 1):
            sel = np.flatnonzero(folds == k)
            for i in range(0, len(sel), batch):
                b = sel[i:i + batch]
                x = torch.from_numpy(feats[b])
                pred = (x[..., :2] + models[k](x)).numpy() * SCALE + origin[b][:, None, :]
                r = rows[b]
                v = r >= 0
                np.add.at(acc, r[v], pred[v])
                np.add.at(cnt, r[v], 1)
    xy = grid[["x", "y"]].to_numpy(float).copy()
    hit = cnt > 0
    xy[hit] = acc[hit] / cnt[hit, None]
    return _finish(grid.assign(x=xy[:, 0], y=xy[:, 1]))
