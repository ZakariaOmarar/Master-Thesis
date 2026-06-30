"""Phase-A fusion forensics — quantify *why* V2 cross-attention fails to help RQ1.

Every prior `INVESTIGATION_CORRECTED.md` settled on the hand-wave
"vibration is peak-stream-limited" without instrumenting the fusion
block.  The 2026-05-15 run undermined that story (V1 vibration NMI
jumped from 0.108 → 0.241), so the failure must live in the fusion
block itself.  This module produces a quantitative, file-backed answer
that lets us choose which Phase-B intervention to run first.

Four measurements, all on the saved V2 encoder (no retrain):

  1. **Token-norm ratio** ``||a_tokens||_F / ||v_tokens||_F`` entering
     `BidirectionalCrossAttention`.  If acoustic tokens have 5–10×
     the norm of vibration tokens the dot-product attention weights
     are dominated by acoustic regardless of which modality carries
     mode-discriminative information.  (Mechanism #4 below.)

  2. **Attention-weight concentration** — Shannon entropy of the
     per-head-averaged attention weights for each cross-attention
     direction.  Low entropy ⇒ attention collapses to one or two
     tokens (could be useful peaks OR a degenerate "ignore" pattern);
     high entropy ≈ ``log(N_kv)`` ⇒ near-uniform attention, i.e. the
     cross-modal signal is being averaged out.  Mechanism #1.

  3. **Cross-modal contribution to c_t** — cosine similarity between
     the c_t produced with both modalities and the c_t produced with
     vibration tokens zeroed *after* the per-modality encoder but
     *before* the fusion block.  Cosine ≈ 1 ⇒ vibration has no effect
     on c_t; cosine << 1 ⇒ vibration *does* shape c_t (whether for
     better or worse is the cluster-purity question).  Mechanism #1
     + #2.

  4. **Gradient magnitude** — ``||d||c_t||² / dac_input||``
     vs ``||d||c_t||² / dvib_input||`` averaged over a held-out batch.
     Direct counterfactual: which input modality moves c_t more under
     small perturbations.  Mechanism #1 quantified differently from
     (3) — this catches the case where small perturbations affect
     c_t but the binary zero-out doesn't (saturation).

  5. **LMM cross-modal dependence** — per-modality LMM cosine loss,
     re-evaluated under a counterfactual where the *other* modality's
     K/V is zeroed at the fusion block.  If LMM is identical with or
     without cross-modal K/V, the reconstruction supervision provides
     no pressure on fusion to mix modalities (mechanism #3).

The module is read-only on the trained V2: it loads weights, runs
forward passes, and writes a JSON + markdown report.  It is **not**
called from `full_run.py` — invoke explicitly as a one-shot::

    python -m src.modeling.eval.fusion_forensics

Outputs (paths fixed for the current run; edit `RUN_DIR` to retarget):

  * ``results/fusion_forensics_v2_20260515.json``
  * ``results/fusion_forensics_v2_20260515.md``
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud

from ...ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _gather_paired_segments,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
)

REPO = Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "results" / "runs" / "20260515_064625__full_seed42"
V2_WEIGHTS = RUN_DIR / "v2" / "encoder.pt"

# Output paths.
OUT_JSON = REPO / "results" / "fusion_forensics_v2_20260515.json"
OUT_MD = REPO / "results" / "fusion_forensics_v2_20260515.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loader(name: str) -> TestDatasetLoader:
    # DatasetSpec.from_yaml resolves all paths to absolute — no reconstruction.
    spec = DatasetSpec.from_yaml(REPO / "configs" / "datasets" / f"{name}.yaml")
    return TestDatasetLoader(spec)


def v2_config() -> V2SSLConfig:
    """Mirror `full_run.v2_config(quick=False)` so we feed the encoder the same
    feature pipeline it was trained on.  Any drift here would invalidate the
    diagnostic."""
    return V2SSLConfig(
        window_seconds=2.0, window_stride_seconds=1.0, feature_dim=64, embed_dim=64,
        n_heads=4, proj_dim=32, epochs=12, batch_size=16, lr=1e-3, weight_decay=1e-5,
        temperature=0.1, val_ratio=0.3,
        # n_mels / n_fft / hop_length inherited from ACOUSTIC_FEATURES
        # (n_fft=4096, hop=2048, n_mels=96 per chapter 3 §3.4.2 grid sweep).
        cwt_n_scales=32, use_cwt=True, gain_jitter_db=6.0,
        channel_dropout_p=0.2, spec_augment_freq_mask=6, spec_augment_time_mask=8,
        lmm_mask_p=0.3, lmm_weight=1.0,
        modality_dropout_p=0.0, acoustic_dropout_p=0.0, vibration_dropout_p=0.5,
        cma_weight=0.0, cma_temperature=0.1,
        context_mode="joint_pma", num_context_seeds=2, seed=42,
    )


def _entropy_per_row(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy (natural log) along the last axis of a probability tensor.

    Shape: ``(*, N_kv) -> (*,)``.  Uniform over N_kv ⇒ ``log(N_kv)``;
    one-hot ⇒ ``0``.
    """
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


@dataclass
class TokenNormStats:
    """`||·||_F` of per-modality token sequences entering the fusion block,
    aggregated across windows.  All values are scalars; the ratio is the
    headline (acoustic_norm / vibration_norm)."""

    n_windows: int
    acoustic_mean_norm: float
    vibration_mean_norm: float
    acoustic_to_vibration_norm_ratio: float
    acoustic_to_vibration_norm_ratio_p50: float
    acoustic_to_vibration_norm_ratio_p95: float


@dataclass
class AttentionConcentrationStats:
    """Shannon entropy of cross-attention weights in each direction, averaged
    over the val batch.  Reported alongside the chance-level entropy
    ``log(N_kv)`` so the reader can tell concentration from chance.

    A ratio entropy / log(N_kv) close to 1.0 means "near-uniform — the cross-
    attention is averaging across everything, which is the same as ignoring
    cross-modal structure."  A ratio close to 0 means "concentrated on one
    or two tokens — could be useful peaks OR a learnt degenerate pattern."
    """

    n_windows: int
    a_from_v_entropy_mean: float
    a_from_v_entropy_log_chance: float
    a_from_v_entropy_ratio_to_chance: float
    v_from_a_entropy_mean: float
    v_from_a_entropy_log_chance: float
    v_from_a_entropy_ratio_to_chance: float


@dataclass
class CrossModalCtStats:
    """Cosine similarity between the full c_t and the c_t produced with one
    modality's tokens zeroed *after* per-modality encoding but *before*
    fusion.  Captures the cross-modal contribution at the c_t level.

    Cosine 1.0 ⇒ removing that modality has no effect ⇒ fusion ignores it.
    Cosine 0.0 ⇒ removing that modality flips c_t entirely ⇒ fusion fully
    depends on it.  Typical healthy multimodal fusion: 0.6–0.8.
    """

    n_windows: int
    cosine_ct_full_vs_zero_vib_mean: float
    cosine_ct_full_vs_zero_vib_p50: float
    cosine_ct_full_vs_zero_vib_p05: float
    cosine_ct_full_vs_zero_ac_mean: float
    cosine_ct_full_vs_zero_ac_p50: float
    cosine_ct_full_vs_zero_ac_p05: float


@dataclass
class GradientStats:
    """``||d||c_t||²/dx_input||`` averaged over the val batch, for each
    modality.  Compares how strongly small input perturbations move c_t.
    A modality with gradient ≈ 0 has effectively been frozen out of c_t.
    """

    n_windows: int
    grad_norm_acoustic_mean: float
    grad_norm_vibration_mean: float
    grad_norm_ratio_acoustic_to_vibration: float


@dataclass
class LMMCrossModalStats:
    """Per-modality LMM cosine loss, reported in two regimes:

    * ``with_xmodal_kv`` — the default trained behaviour: vibration K/V
      available when reconstructing masked acoustic tokens (and vice
      versa).
    * ``without_xmodal_kv`` — counterfactual: the *other* modality's
      K/V is zeroed at the fusion block; the masked-token predictor
      must reconstruct from same-modality context only.

    If the two losses are identical, the LMM head is not using the
    cross-modal channel at all — mechanism #3 confirmed.  If
    ``without_xmodal_kv`` is markedly larger, cross-modal info *is*
    being used and the failure is downstream (the c_t pool either
    discards it or it is dominated by acoustic).
    """

    n_windows: int
    lmm_a_with_xmodal: float
    lmm_a_without_xmodal: float
    lmm_v_with_xmodal: float
    lmm_v_without_xmodal: float


# ---------------------------------------------------------------------------
# Measurement passes
# ---------------------------------------------------------------------------


def _load_v2_encoder(weights_path: Path, cfg: V2SSLConfig) -> V2FusionEncoder:
    enc = V2FusionEncoder(
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        context_mode=cfg.context_mode,
        num_context_seeds=cfg.num_context_seeds,
    )
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[forensics] state_dict load: missing={missing} unexpected={unexpected}")
    enc.eval()
    return enc


def _build_val_loader(
    segments: Iterable[_PairedSegment], cfg: V2SSLConfig
) -> tud.DataLoader:
    ds = _PairedWindowedDataset(list(segments), cfg)
    return tud.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(
            ds, cfg.batch_size, shuffle=False, seed=cfg.seed
        ),
        collate_fn=_collate,
    )


def _measure_token_norms_and_attention(
    encoder: V2FusionEncoder, loader: tud.DataLoader
) -> tuple[TokenNormStats, AttentionConcentrationStats, CrossModalCtStats]:
    a_norms: list[float] = []
    v_norms: list[float] = []
    a_from_v_entropies: list[float] = []
    v_from_a_entropies: list[float] = []
    n_a_kv: list[int] = []
    n_v_kv: list[int] = []
    cos_full_zero_vib: list[float] = []
    cos_full_zero_ac: list[float] = []

    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"]
            vib = batch["vib_feat"]
            ac_xyz = batch["ac_xyz"]
            vib_xyz = batch["vib_xyz"]
            ds_idx = batch["dataset_idx"]

            a_tokens, v_tokens, a_summary, v_summary = (
                encoder.encode_modalities_with_summaries(
                    ac, ac_xyz, vib, vib_xyz, ds_idx
                )
            )
            # Per-window Frobenius norms.
            a_norms.extend(
                a_tokens.float().pow(2).sum(dim=(1, 2)).sqrt().tolist()
            )
            v_norms.extend(
                v_tokens.float().pow(2).sum(dim=(1, 2)).sqrt().tolist()
            )

            # Cross-attention with weights.
            _, _, attn_a_from_v, attn_v_from_a = (
                encoder.fusion.forward_with_attn(a_tokens, v_tokens)
            )
            # attn shape (B, N_q, N_kv); entropy along N_kv averaged over (B, N_q).
            ent_av = _entropy_per_row(attn_a_from_v).mean(dim=1)  # (B,)
            ent_va = _entropy_per_row(attn_v_from_a).mean(dim=1)
            a_from_v_entropies.extend(ent_av.tolist())
            v_from_a_entropies.extend(ent_va.tolist())
            n_a_kv.append(int(a_tokens.shape[1]))
            n_v_kv.append(int(v_tokens.shape[1]))

            # c_t with and without each modality's tokens at fusion input.
            zeros_a = torch.zeros_like(a_tokens)
            zeros_v = torch.zeros_like(v_tokens)
            _, _, ct_full = encoder.fuse_and_pool(
                a_tokens, v_tokens, a_summary=a_summary, v_summary=v_summary
            )
            _, _, ct_zero_vib = encoder.fuse_and_pool(
                a_tokens, zeros_v, a_summary=a_summary, v_summary=v_summary
            )
            _, _, ct_zero_ac = encoder.fuse_and_pool(
                zeros_a, v_tokens, a_summary=a_summary, v_summary=v_summary
            )
            cos_full_zero_vib.extend(
                F.cosine_similarity(ct_full, ct_zero_vib, dim=-1).tolist()
            )
            cos_full_zero_ac.extend(
                F.cosine_similarity(ct_full, ct_zero_ac, dim=-1).tolist()
            )

    a_arr = np.asarray(a_norms)
    v_arr = np.asarray(v_norms)
    ratio = a_arr / np.clip(v_arr, 1e-9, None)
    n_a = int(np.median(n_a_kv)) if n_a_kv else 0
    n_v = int(np.median(n_v_kv)) if n_v_kv else 0
    log_chance_av = float(np.log(max(n_v, 1)))
    log_chance_va = float(np.log(max(n_a, 1)))

    norms = TokenNormStats(
        n_windows=len(a_norms),
        acoustic_mean_norm=float(a_arr.mean()),
        vibration_mean_norm=float(v_arr.mean()),
        acoustic_to_vibration_norm_ratio=float(ratio.mean()),
        acoustic_to_vibration_norm_ratio_p50=float(np.percentile(ratio, 50)),
        acoustic_to_vibration_norm_ratio_p95=float(np.percentile(ratio, 95)),
    )
    attn = AttentionConcentrationStats(
        n_windows=len(a_from_v_entropies),
        a_from_v_entropy_mean=float(np.mean(a_from_v_entropies)),
        a_from_v_entropy_log_chance=log_chance_av,
        a_from_v_entropy_ratio_to_chance=(
            float(np.mean(a_from_v_entropies) / max(log_chance_av, 1e-9))
        ),
        v_from_a_entropy_mean=float(np.mean(v_from_a_entropies)),
        v_from_a_entropy_log_chance=log_chance_va,
        v_from_a_entropy_ratio_to_chance=(
            float(np.mean(v_from_a_entropies) / max(log_chance_va, 1e-9))
        ),
    )
    cos_v = np.asarray(cos_full_zero_vib)
    cos_a = np.asarray(cos_full_zero_ac)
    cstats = CrossModalCtStats(
        n_windows=len(cos_full_zero_vib),
        cosine_ct_full_vs_zero_vib_mean=float(cos_v.mean()),
        cosine_ct_full_vs_zero_vib_p50=float(np.percentile(cos_v, 50)),
        cosine_ct_full_vs_zero_vib_p05=float(np.percentile(cos_v, 5)),
        cosine_ct_full_vs_zero_ac_mean=float(cos_a.mean()),
        cosine_ct_full_vs_zero_ac_p50=float(np.percentile(cos_a, 50)),
        cosine_ct_full_vs_zero_ac_p05=float(np.percentile(cos_a, 5)),
    )
    return norms, attn, cstats


def _measure_input_gradients(
    encoder: V2FusionEncoder, loader: tud.DataLoader, max_batches: int = 5
) -> GradientStats:
    """Compute ``||d||c_t||² / dx||`` for each modality on a small subset.

    Limited to ``max_batches`` because each call needs a fresh backward
    pass.  The encoder is in eval mode but gradients are enabled for the
    *inputs* only.
    """
    grad_a_norms: list[float] = []
    grad_v_norms: list[float] = []
    n_seen = 0
    for batch in loader:
        if n_seen >= max_batches:
            break
        ac = batch["ac_feat"].detach().requires_grad_(True)
        vib = batch["vib_feat"].detach().requires_grad_(True)
        out = encoder(
            ac, batch["ac_xyz"], vib, batch["vib_xyz"], batch["dataset_idx"],
            mask_p=0.0,
        )
        # Use ||c_t||² as the scalar so the gradient is well-defined.
        scalar = (out["context"] ** 2).sum()
        grad_ac, grad_vib = torch.autograd.grad(
            scalar, [ac, vib], retain_graph=False, create_graph=False
        )
        # Per-window gradient norms (flatten over all non-batch dims).
        g_a = grad_ac.detach().flatten(1).pow(2).sum(dim=1).sqrt()
        g_v = grad_vib.detach().flatten(1).pow(2).sum(dim=1).sqrt()
        grad_a_norms.extend(g_a.tolist())
        grad_v_norms.extend(g_v.tolist())
        n_seen += 1

    ga = np.asarray(grad_a_norms)
    gv = np.asarray(grad_v_norms)
    return GradientStats(
        n_windows=int(ga.size),
        grad_norm_acoustic_mean=float(ga.mean()) if ga.size else float("nan"),
        grad_norm_vibration_mean=float(gv.mean()) if gv.size else float("nan"),
        grad_norm_ratio_acoustic_to_vibration=(
            float(ga.mean() / max(gv.mean(), 1e-12)) if ga.size and gv.size else float("nan")
        ),
    )


def _measure_lmm_crossmodal(
    encoder: V2FusionEncoder, loader: tud.DataLoader, cfg: V2SSLConfig
) -> LMMCrossModalStats:
    """Per-modality LMM cosine loss with and without cross-modal K/V at fusion.

    "Without cross-modal K/V" means: zero the *other* modality's input
    tokens before the cross-attention block, then run the LMM cosine
    against the same pre-mask targets.  If the model never relied on
    cross-modal context, the two LMM losses are identical.
    """
    lmm_a_with: list[float] = []
    lmm_a_without: list[float] = []
    lmm_v_with: list[float] = []
    lmm_v_without: list[float] = []
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    with torch.no_grad():
        for batch in loader:
            a_target, v_target, a_summary, v_summary = (
                encoder.encode_modalities_with_summaries(
                    batch["ac_feat"], batch["ac_xyz"],
                    batch["vib_feat"], batch["vib_xyz"], batch["dataset_idx"],
                )
            )
            # Sample masks once; reuse across the with/without comparisons so
            # the only thing varying is whether cross-modal K/V is present.
            a_input, mask_a = encoder.apply_mask(a_target, cfg.lmm_mask_p, gen)
            v_input, mask_v = encoder.apply_mask(v_target, cfg.lmm_mask_p, gen)

            # WITH cross-modal K/V.
            fused_a_w, fused_v_w, _ = encoder.fuse_and_pool(
                a_input, v_input, a_summary=a_summary, v_summary=v_summary
            )
            # WITHOUT cross-modal K/V (for the *acoustic* reconstruction,
            # zero the vibration input; for the *vibration* reconstruction,
            # zero the acoustic input — separate passes).
            fused_a_wo, _, _ = encoder.fuse_and_pool(
                a_input, torch.zeros_like(v_input),
                a_summary=a_summary, v_summary=v_summary,
            )
            _, fused_v_wo, _ = encoder.fuse_and_pool(
                torch.zeros_like(a_input), v_input,
                a_summary=a_summary, v_summary=v_summary,
            )

            def _cos_loss(fused: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
                if not mask.any():
                    return float("nan")
                p = F.normalize(fused[mask], dim=-1)
                t = F.normalize(target[mask], dim=-1)
                return float((1.0 - (p * t).sum(-1)).mean().item())

            lmm_a_with.append(_cos_loss(fused_a_w, a_target, mask_a))
            lmm_a_without.append(_cos_loss(fused_a_wo, a_target, mask_a))
            lmm_v_with.append(_cos_loss(fused_v_w, v_target, mask_v))
            lmm_v_without.append(_cos_loss(fused_v_wo, v_target, mask_v))

    def _nanmean(xs: list[float]) -> float:
        arr = np.asarray([x for x in xs if not np.isnan(x)])
        return float(arr.mean()) if arr.size else float("nan")

    return LMMCrossModalStats(
        n_windows=len(lmm_a_with),
        lmm_a_with_xmodal=_nanmean(lmm_a_with),
        lmm_a_without_xmodal=_nanmean(lmm_a_without),
        lmm_v_with_xmodal=_nanmean(lmm_v_with),
        lmm_v_without_xmodal=_nanmean(lmm_v_without),
    )


# ---------------------------------------------------------------------------
# Verdict + report
# ---------------------------------------------------------------------------


def _verdict(
    norms: TokenNormStats,
    attn: AttentionConcentrationStats,
    ct: CrossModalCtStats,
    grads: GradientStats,
    lmm: LMMCrossModalStats,
) -> tuple[str, list[str]]:
    """Pick the dominant failure mechanism among #1–#4 (instrumented by
    measurements 1–5) and return a one-line verdict plus a bullet list of
    recommended Phase-B interventions in priority order.

    The decision rules are intentionally simple and explicit so they can be
    audited directly in the report.
    """
    bullets: list[str] = []
    # Mechanism #1: residual shortcut / ignore vibration.
    vibration_ignored = (
        ct.cosine_ct_full_vs_zero_vib_mean > 0.95
        and grads.grad_norm_ratio_acoustic_to_vibration > 5.0
    )
    # Mechanism #4: token-norm dominance.
    acoustic_dominates_by_norm = norms.acoustic_to_vibration_norm_ratio_p50 > 2.0
    # Mechanism #1 variant: attention near-uniform = averaging out cross-modal.
    attention_diffused = (
        attn.a_from_v_entropy_ratio_to_chance > 0.95
        or attn.v_from_a_entropy_ratio_to_chance > 0.95
    )
    # Mechanism #3: LMM doesn't use cross-modal K/V.
    lmm_indifferent_to_xmodal = (
        abs(lmm.lmm_a_with_xmodal - lmm.lmm_a_without_xmodal) < 0.01
        and abs(lmm.lmm_v_with_xmodal - lmm.lmm_v_without_xmodal) < 0.01
    )

    if vibration_ignored:
        verdict = (
            "Mechanism #1 dominant — fusion has learned to ignore vibration: "
            f"cos(c_t, c_t|vib=0)={ct.cosine_ct_full_vs_zero_vib_mean:.3f}, "
            f"||dc_t/dac||/||dc_t/dvib||="
            f"{grads.grad_norm_ratio_acoustic_to_vibration:.2f}."
        )
        bullets.append("B1 (symmetric modality dropout) — undoes the self-fulfilling asymmetry.")
        bullets.append("B4 (learnable residual scaling α) — penalises the ignore-vibration shortcut.")
    elif acoustic_dominates_by_norm:
        verdict = (
            "Mechanism #4 dominant — acoustic tokens dwarf vibration tokens at "
            f"fusion input (median norm ratio {norms.acoustic_to_vibration_norm_ratio_p50:.2f}); "
            "dot-product attention is biased toward acoustic regardless of content."
        )
        bullets.append("B2 (pre-fusion LayerNorm) — equalise token scales.")
        bullets.append("B1 (symmetric dropout) — secondary, addresses training-signal asymmetry.")
    elif attention_diffused:
        verdict = (
            "Mechanism #1 (diffused) — cross-attention weights are near-uniform "
            f"(a→v entropy ratio {attn.a_from_v_entropy_ratio_to_chance:.2f}, "
            f"v→a ratio {attn.v_from_a_entropy_ratio_to_chance:.2f}); cross-modal "
            "information is averaged out rather than selectively used."
        )
        bullets.append("B3 (cross-modal LMM) — forces fusion to use the other modality's tokens.")
        bullets.append("B4 (learnable residual scaling) — damps the always-on residual.")
    elif lmm_indifferent_to_xmodal:
        verdict = (
            "Mechanism #3 dominant — LMM loss is indifferent to cross-modal K/V "
            f"(Δ_a={lmm.lmm_a_with_xmodal - lmm.lmm_a_without_xmodal:+.4f}, "
            f"Δ_v={lmm.lmm_v_with_xmodal - lmm.lmm_v_without_xmodal:+.4f}); "
            "the SSL objective provides no pressure for fusion to mix modalities."
        )
        bullets.append("B3 (cross-modal LMM) — directly fixes the SSL pressure mismatch.")
        bullets.append("B5 (CMA on) — adds explicit cross-modal alignment loss.")
    else:
        verdict = (
            "No single mechanism dominates — fusion *does* use both modalities "
            f"(cos(c_t, c_t|vib=0)={ct.cosine_ct_full_vs_zero_vib_mean:.3f}, "
            f"grad ratio {grads.grad_norm_ratio_acoustic_to_vibration:.2f}) but "
            "the fused c_t is no better than acoustic-only at cluster discrimination. "
            "The failure is downstream of fusion (PMA bottleneck or pool-level averaging)."
        )
        bullets.append("B2 (pre-fusion LayerNorm) — prerequisite for any other fix.")
        bullets.append("B5 (CMA on) — explicit alignment may regularise the pool.")
        bullets.append("B1 (symmetric dropout) — cheap baseline to confirm the asymmetry isn't load-bearing.")

    return verdict, bullets


def _write_report(
    norms: TokenNormStats,
    attn: AttentionConcentrationStats,
    ct: CrossModalCtStats,
    grads: GradientStats,
    lmm: LMMCrossModalStats,
    verdict: str,
    bullets: list[str],
) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "v2_weights": str(V2_WEIGHTS.relative_to(REPO)),
        "token_norms": asdict(norms),
        "attention_concentration": asdict(attn),
        "ct_cross_modal_contribution": asdict(ct),
        "input_gradients": asdict(grads),
        "lmm_cross_modal_dependence": asdict(lmm),
        "verdict": verdict,
        "phase_b_priority": bullets,
        "method": "v2_encoder_forensics_2026_05_15",
    }
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines: list[str] = []
    lines.append("# V2 fusion forensics — 2026-05-15 archived encoder\n")
    lines.append(f"**Verdict.** {verdict}\n")
    lines.append("**Recommended Phase-B order:**")
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Measurements\n")
    lines.append("### 1. Token norms entering cross-attention")
    lines.append(f"- ||a_tokens||_F mean = {norms.acoustic_mean_norm:.3f}")
    lines.append(f"- ||v_tokens||_F mean = {norms.vibration_mean_norm:.3f}")
    lines.append(
        f"- ratio (acoustic / vibration) — mean {norms.acoustic_to_vibration_norm_ratio:.3f}, "
        f"p50 {norms.acoustic_to_vibration_norm_ratio_p50:.3f}, "
        f"p95 {norms.acoustic_to_vibration_norm_ratio_p95:.3f}"
    )
    lines.append("")
    lines.append("### 2. Cross-attention concentration (entropy / log(N_kv))")
    lines.append(
        f"- a→v: H={attn.a_from_v_entropy_mean:.3f}, "
        f"chance log(N_v)={attn.a_from_v_entropy_log_chance:.3f}, "
        f"ratio {attn.a_from_v_entropy_ratio_to_chance:.3f}"
    )
    lines.append(
        f"- v→a: H={attn.v_from_a_entropy_mean:.3f}, "
        f"chance log(N_a)={attn.v_from_a_entropy_log_chance:.3f}, "
        f"ratio {attn.v_from_a_entropy_ratio_to_chance:.3f}"
    )
    lines.append("")
    lines.append("### 3. cos(c_t, c_t | one modality zeroed at fusion input)")
    lines.append(
        f"- vibration zeroed: mean {ct.cosine_ct_full_vs_zero_vib_mean:.4f}, "
        f"p50 {ct.cosine_ct_full_vs_zero_vib_p50:.4f}, "
        f"p05 {ct.cosine_ct_full_vs_zero_vib_p05:.4f}"
    )
    lines.append(
        f"- acoustic zeroed: mean {ct.cosine_ct_full_vs_zero_ac_mean:.4f}, "
        f"p50 {ct.cosine_ct_full_vs_zero_ac_p50:.4f}, "
        f"p05 {ct.cosine_ct_full_vs_zero_ac_p05:.4f}"
    )
    lines.append("")
    lines.append("### 4. Input-gradient norms ||d||c_t||²/dx||")
    lines.append(f"- acoustic mean: {grads.grad_norm_acoustic_mean:.4f}")
    lines.append(f"- vibration mean: {grads.grad_norm_vibration_mean:.4f}")
    lines.append(
        f"- ratio (acoustic / vibration): {grads.grad_norm_ratio_acoustic_to_vibration:.2f}"
    )
    lines.append("")
    lines.append("### 5. LMM loss with vs without cross-modal K/V")
    lines.append(
        f"- acoustic reconstruction: with xmodal = {lmm.lmm_a_with_xmodal:.4f}, "
        f"without = {lmm.lmm_a_without_xmodal:.4f}, "
        f"Δ = {lmm.lmm_a_with_xmodal - lmm.lmm_a_without_xmodal:+.4f}"
    )
    lines.append(
        f"- vibration reconstruction: with xmodal = {lmm.lmm_v_with_xmodal:.4f}, "
        f"without = {lmm.lmm_v_without_xmodal:.4f}, "
        f"Δ = {lmm.lmm_v_with_xmodal - lmm.lmm_v_without_xmodal:+.4f}"
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = v2_config()
    print(f"[forensics] Loading V2 weights from {V2_WEIGHTS} ...")
    encoder = _load_v2_encoder(V2_WEIGHTS, cfg)

    print("[forensics] Gathering D1+D2 healthy paired segments ...")
    loaders = [_loader("d1"), _loader("d2")]
    segments = _gather_paired_segments(loaders, cfg)
    loader = _build_val_loader(segments, cfg)
    print(f"[forensics] {len(segments)} paired segments; running forward passes ...")

    norms, attn, ct = _measure_token_norms_and_attention(encoder, loader)
    print(f"[forensics] token norms / attention / c_t cosines done "
          f"(n_windows={norms.n_windows})")
    grads = _measure_input_gradients(encoder, loader, max_batches=5)
    print(f"[forensics] input gradients done (n_windows={grads.n_windows})")
    lmm = _measure_lmm_crossmodal(encoder, loader, cfg)
    print(f"[forensics] LMM cross-modal ablation done (n_batches={lmm.n_windows})")

    verdict, bullets = _verdict(norms, attn, ct, grads, lmm)
    _write_report(norms, attn, ct, grads, lmm, verdict, bullets)
    print(f"\n[forensics] VERDICT: {verdict}")
    print("[forensics] Recommended Phase-B order:")
    for b in bullets:
        print(f"  - {b}")
    print(f"\n[forensics] Wrote {OUT_JSON.relative_to(REPO)}")
    print(f"[forensics] Wrote {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
