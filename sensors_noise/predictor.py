"""Uncertainty-aware trajectory prediction for proactive conflict detection (step 4).

Model: a GRU reads 2 s (20 frames) of one agent's observed track and predicts the next
3 s (30 frames) of positions relative to its last observed position. An ensemble of
N_ENSEMBLE models (different seeds) gives a prediction and its spread (uncertainty).
No interaction modelling: each agent is predicted on its own.

Two training variants, identical architecture and data volume:
  'clean'        histories and targets from clean trajectories
  'noise_aware'  histories degraded with the step-2 noise mix, clean targets
Two-fold split by agent as for the denoiser; training seeds disjoint from evaluation.

Pair-level evaluation: for every reference conflict (step 1) and an equal number of
hard negatives (closing pairs that came within 2-6 m but never conflicted), take both
agents' histories up to 3 s before the critical moment, predict 3 s ahead, and compute
the predicted minimum footprint gap. Predicted gap <= 1 m means a predicted conflict.
Reported: AUC of predicted gap as a conflict score, precision/recall at 1 m, gap error
against the clean reference, and whether the ensemble spread flags wrong decisions.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except ImportError:
    pass

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .conflicts import detect_conflicts, ttc_conflicts
from .denoiser import CATS, SCALE, agent_fold
from .filters import fill_gaps
from .noise import apply_noise
from .ssm import rect_gap

T_HIST, T_FUT = 20, 30                 # frames (10 Hz): 2 s history, 3 s horizon
N_ENSEMBLE = 5
LEAD_FRAMES = 30                       # anchor the prediction 3 s before the critical moment
NEG_GAP = (2.0, 6.0)                   # realised footprint gap range for hard negatives
TRAIN_SEED_BASE = 5000

NOISE_MIX = {"jitter": [0.0, 0.1, 0.3, 0.5, 1.0], "distance_jitter": [0.3, 0.5],
             "drift": [0.3, 0.5], "offset": [0.3, 0.6, 1.0], "dropout": [0.05, 0.15, 0.30]}


def _torch():
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() // 2)))
    return torch


def build_model(hidden: int = 64):
    torch = _torch()
    nn = torch.nn

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(2 + 1 + len(CATS), hidden)
            self.gru = nn.GRU(hidden, hidden, 2, batch_first=True)
            self.out = nn.Linear(hidden, T_FUT * 2)
            nn.init.zeros_(self.out.weight)      # untrained model predicts "stays put"
            nn.init.zeros_(self.out.bias)

        def forward(self, feats):                # (B, T_HIST, 6) -> (B, T_FUT, 2)
            h = torch.relu(self.inp(feats))
            _, hn = self.gru(h)
            return self.out(hn[-1]).view(-1, T_FUT, 2)

    return Predictor()


# --------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------

def _features(hist_xy, hist_obs, cat_idx):
    """Relative, scaled history features; origin = last history position."""
    origin = hist_xy[:, -1, :]
    rel = (hist_xy - origin[:, None, :]) / SCALE
    onehot = np.eye(len(CATS))[cat_idx]
    feats = np.concatenate([rel, hist_obs[..., None].astype(float),
                            np.repeat(onehot[:, None, :], hist_xy.shape[1], 1)], -1)
    return feats.astype(np.float32), origin


def training_windows(grid: pd.DataFrame, clean_xy: np.ndarray, stride: int = 5):
    """Windows of T_HIST (noisy or clean) history + T_FUT clean future from a gap-filled
    grid aligned with clean positions. Returns feats, targets, folds."""
    xy = grid[["x", "y"]].to_numpy(float)
    obs = grid["observed"].to_numpy(bool)
    cat_idx = grid["category"].map({c: i for i, c in enumerate(CATS)}).fillna(0).astype(int).to_numpy()
    folds = grid["orig_uid"].map(agent_fold).to_numpy()
    H, Tg, F = [], [], []
    L = T_HIST + T_FUT
    for idx in grid.groupby("uid", sort=False).indices.values():
        if len(idx) < L:
            continue
        for s in range(0, len(idx) - L + 1, stride):
            seg = idx[s:s + L]
            if obs[seg[:T_HIST]].sum() < 5:
                continue
            H.append(seg[:T_HIST]); Tg.append(seg[T_HIST:]); F.append(folds[seg[0]])
    if not H:
        return (np.zeros((0, T_HIST, 6), np.float32), np.zeros((0, T_FUT, 2), np.float32),
                np.zeros(0, int))
    H, Tg = np.stack(H), np.stack(Tg)
    feats, origin = _features(xy[H], obs[H], cat_idx[H[:, 0]])
    targets = ((clean_xy[Tg] - origin[:, None, :]) / SCALE).astype(np.float32)
    return feats, targets, np.array(F)


def build_training_set(recordings, variant: str, total_windows: int = 200_000,
                       verbose: bool = True):
    conds = ([("jitter", 0.0)] * sum(len(v) for v in NOISE_MIX.values()) if variant == "clean"
             else [(k, lv) for k, levels in NOISE_MIX.items() for lv in levels])
    copies = [(ri, k, lv) for ri in range(len(recordings)) for k, lv in conds]
    per_copy = int(np.ceil(total_windows / len(copies)))
    rng = np.random.default_rng(TRAIN_SEED_BASE)
    F, T, Fo = [], [], []
    for ci, (ri, kind, level) in enumerate(copies):
        clean = recordings[ri]
        noisy = apply_noise(clean, kind, level, np.random.default_rng(TRAIN_SEED_BASE + 7919 * ci + 1))
        grid = fill_gaps(noisy)
        cxy = clean.set_index(["uid", "tkey"])[["x", "y"]]
        cxy = cxy.reindex(pd.MultiIndex.from_arrays([grid["orig_uid"], grid["tkey"]]))
        ok = cxy["x"].notna().to_numpy()
        grid, cxy = grid[ok].reset_index(drop=True), cxy[ok].to_numpy(float)
        feats, targets, folds = training_windows(grid, cxy)
        sel = np.arange(len(feats))
        if len(sel) > per_copy:
            sel = rng.choice(sel, per_copy, replace=False)
        F.append(feats[sel]); T.append(targets[sel]); Fo.append(folds[sel])
        if verbose:
            print(f"  [{ci + 1}/{len(copies)}] rec {ri} {kind} {level:g}: {len(sel)} windows")
    return np.concatenate(F), np.concatenate(T), np.concatenate(Fo)


def train_model(feats, targets, epochs: int = 8, batch: int = 256, lr: float = 1e-3,
                device: str = "auto", seed: int = 0, verbose: bool = True):
    torch = _torch()
    torch.manual_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model = build_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    X, Y = torch.from_numpy(feats), torch.from_numpy(targets)
    huber = torch.nn.SmoothL1Loss(reduction="none", beta=0.05)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            x, y = X[b].to(dev), Y[b].to(dev)
            p = model(x)
            lp = huber(p, y).mean()
            lv = huber(p[:, 1:] - p[:, :-1], y[:, 1:] - y[:, :-1]).mean()
            loss = lp + 10.0 * lv
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if verbose:
            print(f"      epoch {ep + 1}/{epochs}: loss {tot / n:.5f}")
    return model.cpu()


def train_variant(recordings, variant: str, out_dir, total_windows: int = 200_000,
                  epochs: int = 8, device: str = "auto", verbose: bool = True):
    """2 folds x N_ENSEMBLE members, saved as <variant>_fold<k>_m<j>.pt."""
    torch = _torch()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Building training windows for predictor variant '{variant}'")
    feats, targets, folds = build_training_set(recordings, variant, total_windows, verbose)
    for k in (0, 1):
        sel = folds != k
        for j in range(N_ENSEMBLE):
            if verbose:
                print(f"  training {variant} fold {k} member {j} on {sel.sum()} windows")
            m = train_model(feats[sel], targets[sel], epochs=epochs, device=device,
                            seed=100 * k + j, verbose=verbose)
            torch.save({"state_dict": m.state_dict(), "variant": variant, "fold": k, "member": j},
                       out / f"{variant}_fold{k}_m{j}.pt")
    return out


_CACHE = {}


def load_ensembles(models_dir, variant: str):
    torch = _torch()
    key = (str(models_dir), variant)
    if key not in _CACHE:
        ens = []
        for k in (0, 1):
            ms = []
            for j in range(N_ENSEMBLE):
                ck = torch.load(Path(models_dir) / f"{variant}_fold{k}_m{j}.pt",
                                map_location="cpu", weights_only=False)
                m = build_model(); m.load_state_dict(ck["state_dict"]); m.eval(); ms.append(m)
            ens.append(ms)
        _CACHE[key] = ens
    return _CACHE[key]


# --------------------------------------------------------------------------------------
# Pair-level evaluation
# --------------------------------------------------------------------------------------

def reference_pairs(clean_df: pd.DataFrame, ttc_threshold: float = 3.0, pet_threshold: float = 2.0,
                    seed: int = 0) -> pd.DataFrame:
    """Positives (reference conflicts with their critical time) and an equal number of
    hard negatives from the same screens (closing pairs with realised gap in NEG_GAP
    that are not conflicts; PET pairs with PET in 2-6 s)."""
    base = apply_noise(clean_df, "baseline", 0.0, np.random.default_rng(0))
    pt, pp, static = detect_conflicts(base)
    rows = []
    if len(pt):
        pt = pt[pt["plausible"].astype(bool)]
        pos = ttc_conflicts(pt, ttc_threshold)
        for r in pos.itertuples():
            rows.append((r.uid_a, r.uid_b, int(r.tkey_at_min), 1, "TTC", r.interaction))
        neg = pt[(~pt.index.isin(pos.index)) & pt["min_gap"].between(*NEG_GAP)]
        for r in neg.itertuples():
            rows.append((r.uid_a, r.uid_b, int(r.tkey_at_min), 0, "TTC", r.interaction))
    if len(pp):
        pp = pp[pp["plausible"].astype(bool)]
        for r in pp.itertuples():
            lab = 1 if r.pet < pet_threshold else (0 if 2.0 <= r.pet <= 6.0 else -1)
            if lab >= 0:
                rows.append((r.uid_a, r.uid_b, int(r.tkey_first), lab, "PET", r.interaction))
    df = pd.DataFrame(rows, columns=["uid_a", "uid_b", "tkey_crit", "label", "source", "interaction"])
    df = df.drop_duplicates(["uid_a", "uid_b", "source"])
    rng = np.random.default_rng(seed)
    out = []
    for src, g in df.groupby("source"):
        p, n = g[g["label"] == 1], g[g["label"] == 0]
        if len(n) > len(p):
            n = n.iloc[rng.choice(len(n), len(p), replace=False)]
        out += [p, n]
    return pd.concat(out, ignore_index=True)


def _agent_tracks(df: pd.DataFrame):
    g = fill_gaps(df)
    cols = ["tkey", "x", "y", "heading", "observed"]
    tracks = {}
    attrs = g.groupby("uid")[["category", "length", "width", "orig_uid"]].first()
    for uid, idx in g.groupby("uid", sort=False).indices.items():
        tracks[uid] = g.iloc[idx][cols].to_numpy(float)
    return tracks, attrs


def _history(track, t_anchor_ms):
    """Last T_HIST grid rows ending at or before the anchor; None if too short."""
    tk = track[:, 0]
    i = np.searchsorted(tk, t_anchor_ms, side="right") - 1
    if i + 1 < T_HIST:
        return None
    return track[i + 1 - T_HIST:i + 1]


def _fragments_of(tracks, attrs, orig_uid):
    """Track fragments (after ID switches) belonging to one original agent."""
    return [u for u in attrs.index[attrs["orig_uid"] == orig_uid]]


def predict_pairs(df: pd.DataFrame, pairs: pd.DataFrame, ensembles, clean_tracks, clean_attrs,
                  dt_s: float = 0.1) -> pd.DataFrame:
    """Predict both agents of each pair from histories ending LEAD_FRAMES before the
    critical moment and compute predicted / true minimum footprint gaps over the horizon."""
    torch = _torch()
    tracks, attrs = _agent_tracks(df)
    rows = []
    for r in pairs.itertuples():
        t_anchor = r.tkey_crit - int(LEAD_FRAMES * dt_s * 1000)
        hists, info = [], []
        for uid in (r.uid_a, r.uid_b):
            frag = None
            for u in _fragments_of(tracks, attrs, uid):
                h = _history(tracks[u], t_anchor)
                if h is not None and h[-1, 0] >= t_anchor - 300:   # history reaches the anchor
                    frag = h; break
            if frag is None:
                break
            hists.append(frag); info.append(uid)
        if len(hists) < 2:
            rows.append({**r._asdict(), "pred_gap": np.nan, "pred_gap_std": np.nan,
                         "true_gap": np.nan, "skipped": True})
            continue
        preds = []   # per agent: (N_ENSEMBLE, T_FUT, 2)
        for uid, h in zip(info, hists):
            cat_idx = CATS.index(attrs.loc[uid, "category"]) if attrs.loc[uid, "category"] in CATS else 0
            feats, origin = _features(h[None, :, 1:3], h[None, :, 4].astype(bool), np.array([cat_idx]))
            ens = ensembles[agent_fold(uid)]
            with torch.no_grad():
                x = torch.from_numpy(feats)
                p = np.stack([m(x).numpy()[0] for m in ens]) * SCALE + origin[0]
            preds.append(p)
        la, wa = attrs.loc[info[0], ["length", "width"]].astype(float)
        lb, wb = attrs.loc[info[1], ["length", "width"]].astype(float)

        def min_gap(pa, pb, ha0, hb0):
            da, db = np.diff(pa, axis=0, prepend=pa[:1]), np.diff(pb, axis=0, prepend=pb[:1])
            ha = np.where(np.hypot(*da.T) > 0.05, np.arctan2(da[:, 1], da[:, 0]), ha0)
            hb = np.where(np.hypot(*db.T) > 0.05, np.arctan2(db[:, 1], db[:, 0]), hb0)
            return rect_gap(pa[:, 0], pa[:, 1], ha, np.full(len(pa), la), np.full(len(pa), wa),
                            pb[:, 0], pb[:, 1], hb, np.full(len(pb), lb), np.full(len(pb), wb)).min()

        ha0, hb0 = hists[0][-1, 3], hists[1][-1, 3]
        gaps = np.array([min_gap(preds[0][j], preds[1][j], ha0, hb0) for j in range(N_ENSEMBLE)])
        mean_gap = min_gap(preds[0].mean(0), preds[1].mean(0), ha0, hb0)
        # True gap over the same horizon from the clean reference.
        ta, tb = clean_tracks[r.uid_a], clean_tracks[r.uid_b]
        sel_a = (ta[:, 0] > t_anchor) & (ta[:, 0] <= t_anchor + T_FUT * dt_s * 1000)
        sel_b = (tb[:, 0] > t_anchor) & (tb[:, 0] <= t_anchor + T_FUT * dt_s * 1000)
        common, ia, ib = np.intersect1d(ta[sel_a, 0], tb[sel_b, 0], return_indices=True)
        if len(common):
            A, B = ta[sel_a][ia], tb[sel_b][ib]
            true_gap = rect_gap(A[:, 1], A[:, 2], A[:, 3], np.full(len(A), la), np.full(len(A), wa),
                                B[:, 1], B[:, 2], B[:, 3], np.full(len(B), lb), np.full(len(B), wb)).min()
        else:
            true_gap = np.nan
        rows.append({**r._asdict(), "pred_gap": float(mean_gap), "pred_gap_std": float(gaps.std()),
                     "true_gap": float(true_gap), "skipped": False})
    out = pd.DataFrame(rows)
    return out.drop(columns=["Index"], errors="ignore")


def _auc(score, label):
    """Rank-based AUC (ties averaged); NaN if one class is missing."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n1, n0 = (label == 1).sum(), (label == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(score).rank().to_numpy()
    return (ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate_predictions(pred: pd.DataFrame, gap_threshold: float = 1.0) -> pd.DataFrame:
    """Per source (TTC/PET): detection AUC and precision/recall at the gap threshold,
    gap error on positives, and whether ensemble spread flags wrong decisions."""
    rows = []
    p = pred[~pred["skipped"]]
    for src, g in p.groupby("source"):
        lab = g["label"].to_numpy()
        det = (g["pred_gap"] <= gap_threshold).to_numpy()
        tp, fp, fn = (det & (lab == 1)).sum(), (det & (lab == 0)).sum(), (~det & (lab == 1)).sum()
        wrong = (det != (lab == 1)).astype(int)
        err = (g["pred_gap"] - g["true_gap"]).to_numpy()
        rows.append({
            "source": src, "n_pos": int((lab == 1).sum()), "n_neg": int((lab == 0).sum()),
            "skipped": int(pred[pred["source"] == src]["skipped"].sum()),
            "auc_gap": _auc(-g["pred_gap"], lab),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "gap_err_median_pos": float(np.nanmedian(err[lab == 1])),
            "gap_abs_err_median": float(np.nanmedian(np.abs(err))),
            "unc_auc_wrong": _auc(g["pred_gap_std"], wrong),
            "unc_spearman_abs_err": float(pd.Series(g["pred_gap_std"]).corr(
                pd.Series(np.abs(err)), method="spearman")),
        })
    return pd.DataFrame(rows)
