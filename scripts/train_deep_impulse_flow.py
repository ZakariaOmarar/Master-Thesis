"""Train + evaluate the deep impulse-aware anomaly flow — the production
anomaly detector for the rig.

Architecture (src/modeling/anomaly/deep_impulse_flow.py): a spectrogram 2-D CNN
front-end trained end-to-end one-class with a conditional normalizing flow on
the healthy NLL (no SSL/CMA), with the hand-crafted impulse+spectral anchor
concatenated (recall guarantee + anti-collapse), per-window instance norm
(regime-level invariance), strong regime-simulating augmentation (domain
generalization), early-stop restore-best.  Per modality (mic, accel); anomaly
score = sum of z-scored per-modality NLLs.  Fit on healthy only.  Few-shot
adaptation (--adapt-frac) recovers a brand-new campaign from a little of its
healthy.

This module exposes `collect_features`, `train_and_eval` and `DeepImpulseConfig`
so the hyperparameter search (search_deep_impulse_flow.py) reuses the exact same
pipeline.  Feature extraction is cached on disk (keyed by window/n_mels) so the
search and the multi-seed runs are fast.

Run:
    python -m scripts.train_deep_impulse_flow --epochs 40 --seed 0 --adapt-frac 0.3
    python -m scripts.train_deep_impulse_flow --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.deep_impulse_flow import DeepImpulseConfig, DeepImpulseFlow
from src.modeling.anomaly.raw_impulse_detector import window_features
from src.modeling.anomaly.weak_labels import derive_knock_events, window_overlaps_any

CACHE_DIR = REPO / "results" / "cache"


# ----------------------------------------------------------------------------
# Feature extraction (cached on disk)
# ----------------------------------------------------------------------------
def _spectrogram(w: np.ndarray, fs: float, n_mels: int, n_t: int) -> np.ndarray:
    if w.size < 32 or not np.any(w):
        return np.zeros((n_mels, n_t))
    from scipy.signal import stft
    nper = min(512, max(32, w.size // 16))
    _, _, Z = stft(w.astype(np.float64), fs=fs, nperseg=nper, noverlap=nper // 2)
    mag = np.log1p(np.abs(Z)); F, Tf = mag.shape
    if F >= n_mels:
        mel = np.stack([b.mean(0) for b in np.array_split(mag, n_mels, axis=0)])
    else:
        fi = np.linspace(0, F - 1, n_mels)
        mel = np.stack([np.interp(fi, np.arange(F), mag[:, j]) for j in range(Tf)], axis=1)
    if mel.shape[1] != n_t:
        xi = np.linspace(0, mel.shape[1] - 1, n_t)
        mel = np.stack([np.interp(xi, np.arange(mel.shape[1]), mel[i]) for i in range(n_mels)])
    return (mel - mel.mean()) / (mel.std() + 1e-6)


def _collect(loader, is_anom: bool, cfg: DeepImpulseConfig):
    msp, asp, manc, aanc, env, peak = [], [], [], [], [], []
    for s in loader.list_segments():
        if s.is_anomaly != is_anom:
            continue
        ds = s.segment
        mic = getattr(ds, "mic_data", None)
        if mic is None or not mic.size:
            continue
        fs_m = float(ds.mic_sample_rate)
        acc = getattr(ds, "accel_data", None)
        fs_a = float(getattr(ds, "accel_sample_rate", 0) or 1.0)
        intervals = derive_knock_events(s, max_events=300, noise_floor_mult=3.0) if is_anom else []
        wm = np.sqrt(np.mean(mic.astype(np.float64) ** 2, axis=0))
        wa = (np.sqrt(np.mean(acc.astype(np.float64) ** 2, axis=0))
              if acc is not None and acc.size else None)
        T = wm.size; step = max(1, int(cfg.stride_s * fs_m)); wlen = int(cfg.win_s * fs_m)
        for i0 in range(0, max(1, T - wlen + 1), step):
            t0, t1 = i0 / fs_m, (i0 + wlen) / fs_m
            seg = wm[i0:i0 + wlen]
            msp.append(_spectrogram(seg, fs_m, cfg.n_mels, cfg.n_t)); manc.append(window_features(seg, fs_m))
            peak.append(float(np.abs(seg).max()) if seg.size else 0.0)
            sa = wa[int(t0 * fs_a):int(t1 * fs_a)] if wa is not None else np.zeros(0)
            asp.append(_spectrogram(sa, fs_a, cfg.n_mels, cfg.n_t)); aanc.append(window_features(sa, fs_a))
            env.append(1 if (is_anom and window_overlaps_any(t0, t1, intervals)) else 0)
    return {"msp": np.asarray(msp), "asp": np.asarray(asp), "manc": np.asarray(manc),
            "aanc": np.asarray(aanc), "env": np.asarray(env), "peak": np.asarray(peak)}


def collect_features(datasets, cfg: DeepImpulseConfig, cache_dir: Path = CACHE_DIR):
    """{ds: (H, A)} with on-disk caching keyed by window/n_mels."""
    from src.modeling.eval.rq2_three_paradigm_eval import _loader
    out = {}
    for dn in datasets:
        key = f"deepimp_{dn}_w{cfg.win_s}_s{cfg.stride_s}_m{cfg.n_mels}_t{cfg.n_t}.npz"
        p = cache_dir / key
        if p.exists():
            d = np.load(p)
            H = {k[2:]: d[k] for k in d.files if k.startswith("h_")}
            A = {k[2:]: d[k] for k in d.files if k.startswith("a_")}
        else:
            L = _loader(dn)
            H = _collect(L, False, cfg); A = _collect(L, True, cfg)
            floor = float(np.percentile(H["peak"], 99.5)) if H["peak"].size else float("inf")
            A["ref"] = (A["peak"] > floor).astype(int)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(p, **{f"h_{k}": v for k, v in H.items()},
                     **{f"a_{k}": v for k, v in A.items()})
        out[dn] = (H, A)
        print(f"  {dn}: healthy={H['msp'].shape[0]} anom={A['msp'].shape[0]}"
              f"{' (cached)' if p.exists() else ''}", flush=True)
    return out


# ----------------------------------------------------------------------------
# Augmentation + training
# ----------------------------------------------------------------------------
def _augment(spec: torch.Tensor, strength: str) -> torch.Tensor:
    if strength == "none":
        return spec
    x = spec.clone(); B, F, T = x.shape; dev = x.device
    if strength == "light":
        x = torch.roll(x, int(torch.randint(-T // 8, T // 8 + 1, (1,))), dims=2)
        return x + 0.02 * torch.randn_like(x)
    # strong: domain-generalization augmentation
    x = x * (1.0 + 0.4 * (torch.rand(B, 1, 1, device=dev) - 0.5))
    x = torch.roll(x, int(torch.randint(-F // 6, F // 6 + 1, (1,))), dims=1)
    x = torch.roll(x, int(torch.randint(-T // 6, T // 6 + 1, (1,))), dims=2)
    fm = int(torch.randint(0, F // 4 + 1, (1,))); f0 = int(torch.randint(0, max(1, F - fm), (1,)))
    x[:, f0:f0 + fm, :] = 0.0
    tm = int(torch.randint(0, T // 4 + 1, (1,))); t0 = int(torch.randint(0, max(1, T - tm), (1,)))
    x[:, :, t0:t0 + tm] = 0.0
    return x + 0.05 * torch.randn_like(x)


def _train_one(model, spec, anc, dev, cfg, lr, epochs, patience):
    st = torch.tensor(spec, dtype=torch.float32); at = torch.tensor(anc, dtype=torch.float32)
    n = st.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
    n_val = max(1, int(cfg.val_frac * n)); val_idx, tr_idx = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_val, best_state, bad, best_ep = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train(); tp = tr_idx[torch.randperm(tr_idx.shape[0])]; tot = 0.0
        for i in range(0, tp.shape[0], cfg.batch_size):
            idx = tp[i:i + cfg.batch_size]
            sb = _augment(st[idx].to(dev), cfg.augment); ab = at[idx].to(dev)
            loss = -model.log_prob(sb, ab).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss.detach()) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vt = sum(float(-model.log_prob(st[val_idx[i:i+512]].to(dev),
                                           at[val_idx[i:i+512]].to(dev)).sum())
                     for i in range(0, n_val, 512)) / n_val
        if vt < best_val - 1e-3:
            best_val, bad, best_ep = vt, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 0 or (ep + 1) % 10 == 0 or bad >= patience or ep == epochs - 1:
            print(f"    ep {ep+1}/{epochs} train={tot/tr_idx.shape[0]:.2f} val={vt:.2f} best@{best_ep+1}", flush=True)
        if bad >= patience:
            print(f"    early stop @ {ep+1} (best @ {best_ep+1})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _scores(model, spec, anc, dev):
    model.eval(); out = []
    with torch.no_grad():
        st = torch.tensor(spec, dtype=torch.float32); at = torch.tensor(anc, dtype=torch.float32)
        for i in range(0, st.shape[0], 512):
            out.append(model.anomaly_score(st[i:i+512].to(dev), at[i:i+512].to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ----------------------------------------------------------------------------
# Full train + eval (shared by main and the HP search)
# ----------------------------------------------------------------------------
def train_and_eval(cfg, feats, fit_ds, test_ds, dev, *, do_adapt=False, verbose=True):
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    H = {d: feats[d][0] for d in feats}; A = {d: feats[d][1] for d in feats}
    fm = np.concatenate([H[d]["manc"] for d in fit_ds]); fa = np.concatenate([H[d]["aanc"] for d in fit_ds])
    mmu, msd = fm.mean(0), fm.std(0) + 1e-8; amu, asd = fa.mean(0), fa.std(0) + 1e-8

    mic = DeepImpulseFlow.from_config(cfg).to(dev); acc = DeepImpulseFlow.from_config(cfg).to(dev)
    if verbose:
        print("training mic ...", flush=True)
    _train_one(mic, np.concatenate([H[d]["msp"] for d in fit_ds]), (fm - mmu) / msd,
               dev, cfg, cfg.lr, cfg.epochs, cfg.patience)
    if verbose:
        print("training accel ...", flush=True)
    _train_one(acc, np.concatenate([H[d]["asp"] for d in fit_ds]), (fa - amu) / asd,
               dev, cfg, cfg.lr, cfg.epochs, cfg.patience)

    hzm = _scores(mic, np.concatenate([H[d]["msp"] for d in fit_ds]), (fm - mmu) / msd, dev)
    hza = _scores(acc, np.concatenate([H[d]["asp"] for d in fit_ds]), (fa - amu) / asd, dev)
    mz = (hzm.mean(), hzm.std() + 1e-8); ma = (hza.mean(), hza.std() + 1e-8)

    def _fused(mic_m, acc_m, mz_, ma_, d, sel=slice(None)):
        zm = (_scores(mic_m, d["msp"][sel], (d["manc"][sel] - mmu) / msd, dev) - mz_[0]) / mz_[1]
        za = (_scores(acc_m, d["asp"][sel], (d["aanc"][sel] - amu) / asd, dev) - ma_[0]) / ma_[1]
        return zm + za

    hf = (hzm - mz[0]) / mz[1] + (hza - ma[0]) / ma[1]
    thr = float(np.percentile(hf, 100 * (1 - cfg.target_fpr)))
    res = {"healthy_fpr": float((hf > thr).mean()), "per_dataset": {}}
    for dn in test_ds:
        zh = _fused(mic, acc, mz, ma, H[dn]); za = _fused(mic, acc, mz, ma, A[dn])
        y = np.concatenate([np.zeros(zh.size), np.ones(za.size)]); s = np.concatenate([zh, za])
        yr = A[dn]["ref"]
        res["per_dataset"][dn] = {
            "reclvl_roc": float(roc_auc_score(y, s)), "reclvl_pr": float(average_precision_score(y, s)),
            "href_roc": float(roc_auc_score(yr, za)) if 0 < yr.sum() < yr.size else float("nan"),
            "flag_at_thr": float((za > thr).mean()), "held_out": dn not in fit_ds}

        if do_adapt and dn not in fit_ds and cfg.adapt_frac > 0:
            nH = H[dn]["msp"].shape[0]; idx = np.random.default_rng(cfg.seed).permutation(nH)
            na = max(16, int(cfg.adapt_frac * nH)); ai, ei = idx[:na], idx[na:]
            mA, aA = copy.deepcopy(mic), copy.deepcopy(acc)
            _train_one(mA, H[dn]["msp"][ai], (H[dn]["manc"][ai] - mmu) / msd, dev, cfg,
                       cfg.lr * cfg.adapt_lr_mult, cfg.adapt_epochs, cfg.adapt_epochs)
            _train_one(aA, H[dn]["asp"][ai], (H[dn]["aanc"][ai] - amu) / asd, dev, cfg,
                       cfg.lr * cfg.adapt_lr_mult, cfg.adapt_epochs, cfg.adapt_epochs)
            zmh = _scores(mA, H[dn]["msp"][ai], (H[dn]["manc"][ai] - mmu) / msd, dev)
            zah = _scores(aA, H[dn]["asp"][ai], (H[dn]["aanc"][ai] - amu) / asd, dev)
            mzN = (zmh.mean(), zmh.std() + 1e-8); maN = (zah.mean(), zah.std() + 1e-8)
            thrA = float(np.percentile((zmh - mzN[0]) / mzN[1] + (zah - maN[0]) / maN[1],
                                       100 * (1 - cfg.target_fpr)))
            zh = _fused(mA, aA, mzN, maN, H[dn], ei); za = _fused(mA, aA, mzN, maN, A[dn])
            y = np.concatenate([np.zeros(zh.size), np.ones(za.size)]); s = np.concatenate([zh, za])
            res["per_dataset"][dn]["adapted"] = {
                "reclvl_roc": float(roc_auc_score(y, s)), "reclvl_pr": float(average_precision_score(y, s)),
                "flag_at_thr": float((za > thrA).mean()), "eval_fpr": float((zh > thrA).mean())}
    artifacts = {"mic": mic, "acc": acc, "anchor": (mmu, msd, amu, asd),
                 "score_norm": (mz, ma), "threshold": thr}
    return res, artifacts


def _print_table(res):
    print(f"\nhealthy FPR @ threshold = {res['healthy_fpr']:.3f}\n")
    print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'href_ROC':>9} {'flag@thr':>9} {'heldout':>8}")
    for dn, r in res["per_dataset"].items():
        print(f"{dn:<5} {r['reclvl_roc']:>10.3f} {r['reclvl_pr']:>10.3f} "
              f"{r['href_roc'] if not np.isnan(r['href_roc']) else 0:>9.3f} "
              f"{r['flag_at_thr']:>9.3f} {'YES' if r['held_out'] else '':>8}")
    if any("adapted" in r for r in res["per_dataset"].values()):
        print("\nfew-shot adapted (held-out campaigns):")
        print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'flag@thr':>9} {'eval_FPR':>9}")
        for dn, r in res["per_dataset"].items():
            if "adapted" in r:
                a = r["adapted"]
                print(f"{dn:<5} {a['reclvl_roc']:>10.3f} {a['reclvl_pr']:>10.3f} "
                      f"{a['flag_at_thr']:>9.3f} {a['eval_fpr']:>9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ds", nargs="*", default=["d2", "d3", "d4"])
    ap.add_argument("--test-ds", nargs="*", default=["d2", "d3", "d4", "d5"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--augment", choices=("none", "light", "strong"), default="strong")
    ap.add_argument("--adapt-frac", type=float, default=0.0)
    ap.add_argument("--config", default=None,
                    help="JSON of a DeepImpulseConfig (e.g. the HP-search best); "
                         "--seed/--epochs/--adapt-frac still override it")
    ap.add_argument("--out", default="results/deep_impulse_flow.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")

    if args.smoke:
        cfg = DeepImpulseConfig(epochs=4, patience=2)
        m = DeepImpulseFlow.from_config(cfg).to(dev)
        sp = np.random.randn(96, cfg.n_mels, cfg.n_t); an = np.random.randn(96, cfg.n_anchor)
        _train_one(m, sp, an, dev, cfg, cfg.lr, cfg.epochs, cfg.patience)
        print("smoke scores:", _scores(m, sp[:8], an[:8], dev).shape)
        return 0

    if args.config:
        base = DeepImpulseConfig(**json.loads(Path(args.config).read_text()))
        cfg = replace(base, seed=args.seed, epochs=args.epochs, adapt_frac=args.adapt_frac)
        print(f"loaded config from {args.config}")
    else:
        cfg = DeepImpulseConfig(epochs=args.epochs, seed=args.seed, lr=args.lr,
                                patience=args.patience, augment=args.augment,
                                adapt_frac=args.adapt_frac)
    all_ds = sorted(set(args.fit_ds) | set(args.test_ds))
    print(f"collecting features (cache={CACHE_DIR}) ...", flush=True)
    feats = collect_features(all_ds, cfg)
    res, art = train_and_eval(cfg, feats, args.fit_ds, args.test_ds, dev,
                              do_adapt=args.adapt_frac > 0)
    _print_table(res)
    res["config"] = asdict(cfg); res["fit_ds"] = args.fit_ds

    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    mmu, msd, amu, asd = art["anchor"]
    torch.save({"mic": art["mic"].state_dict(), "acc": art["acc"].state_dict(),
                "anchor_stats": {"mmu": mmu, "msd": msd, "amu": amu, "asd": asd},
                "score_norm": art["score_norm"], "threshold": art["threshold"],
                "config": asdict(cfg)}, out)
    with out.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nsaved -> {out.relative_to(REPO)}\nsaved -> {out.with_suffix('.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
