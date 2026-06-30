"""Feature cache must be result-neutral: a hit returns bytes-identical output,
and any input/param change misses.  Also covers the cluster-collapse diagnostic.
"""

from __future__ import annotations

import numpy as np

from src.features.audio_spectral import compute_encoder_input_stack
from src.features.feature_cache import _make_key, disk_cached_feature
from src.features.vibration_temporal import compute_vibration_input_stack
from src.modeling.context.cluster_metric import cluster_purity_and_nmi


def test_cache_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HYDRO_FEATURE_CACHE_DIR", raising=False)
    calls = {"n": 0}

    @disk_cached_feature
    def f(x):
        calls["n"] += 1
        return x * 2.0

    a = np.ones((3, 4))
    np.testing.assert_array_equal(f(a), a * 2)
    f(a)
    assert calls["n"] == 2  # no caching when env var unset


def test_cache_hit_is_exact_and_skips_recompute(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYDRO_FEATURE_CACHE_DIR", str(tmp_path))
    calls = {"n": 0}

    @disk_cached_feature
    def f(x, *, k=1.0):
        calls["n"] += 1
        return (x * k).astype(np.float32)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 7)).astype(np.float32)
    first = f(a, k=3.0)
    second = f(a, k=3.0)  # cache hit
    assert calls["n"] == 1
    np.testing.assert_array_equal(first, second)  # bytes-identical


def test_cache_misses_on_any_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYDRO_FEATURE_CACHE_DIR", str(tmp_path))
    calls = {"n": 0}

    @disk_cached_feature
    def f(x, *, k=1.0):
        calls["n"] += 1
        return x * k

    a = np.ones((2, 2))
    f(a, k=1.0)
    f(a, k=2.0)          # different param → miss
    f(a * 2, k=1.0)      # different input → miss
    assert calls["n"] == 3


def test_key_sensitive_to_dtype_shape_param_and_source() -> None:
    a = np.ones((4,), dtype=np.float32)

    def K(args: dict, src: str = "src") -> str:
        return _make_key("f", src, args)

    assert K({"x": a}) != K({"x": a.astype(np.float64)})        # dtype matters
    assert K({"x": a}) != K({"x": np.ones((5,), np.float32)})   # shape matters
    assert K({"x": a, "k": 1}) != K({"x": a, "k": 2})           # param matters
    assert K({"x": a, "k": 1}) == K({"x": a, "k": 1})           # stable
    # The source-code hash is part of the key, so editing the extractor (same
    # args) cannot return a stale entry.
    assert K({"x": a}, src="src1") != K({"x": a}, src="src2")


def test_default_and_explicit_args_share_one_entry(tmp_path, monkeypatch) -> None:
    """The CRITICAL result-neutrality property: the key reflects the EFFECTIVE
    arguments (defaults applied), so a value passed as a default and the same
    value passed explicitly hit the SAME entry — and a different value misses.
    Without signature binding, omitting a param whose default later changes
    would silently return stale features."""
    monkeypatch.setenv("HYDRO_FEATURE_CACHE_DIR", str(tmp_path))
    calls = {"n": 0}

    @disk_cached_feature
    def f(x, *, k=3.0):
        calls["n"] += 1
        return (x * k).astype(np.float32)

    a = np.ones((3,), np.float32)
    r_default = f(a)          # uses default k=3.0
    r_explicit = f(a, k=3.0)  # same effective value -> must be a cache HIT
    assert calls["n"] == 1
    np.testing.assert_array_equal(r_default, r_explicit)
    f(a, k=4.0)               # different value -> miss
    assert calls["n"] == 2


def test_two_functions_do_not_collide(tmp_path, monkeypatch) -> None:
    """Same signature + same args but different bodies must not share an entry
    (qualname + source hash differ)."""
    monkeypatch.setenv("HYDRO_FEATURE_CACHE_DIR", str(tmp_path))

    @disk_cached_feature
    def g(x):
        return x + 1.0

    @disk_cached_feature
    def h(x):
        return x + 2.0

    a = np.zeros((2,), np.float64)
    np.testing.assert_array_equal(g(a), a + 1.0)
    np.testing.assert_array_equal(h(a), a + 2.0)  # not g's cached result


def test_real_extractors_cache_matches_uncached(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(1)
    mic = rng.standard_normal((2, 8000)).astype(np.float64)
    accel = rng.standard_normal((3, 400)).astype(np.float64)

    monkeypatch.delenv("HYDRO_FEATURE_CACHE_DIR", raising=False)
    ac_ref = compute_encoder_input_stack(mic, fs=16000, n_mels=16, n_fft=256,
                                         hop_length=128, cwt_n_scales=8)
    vib_ref = compute_vibration_input_stack(accel, sample_rate=376.0)

    monkeypatch.setenv("HYDRO_FEATURE_CACHE_DIR", str(tmp_path))
    for _ in range(2):  # miss then hit
        ac = compute_encoder_input_stack(mic, fs=16000, n_mels=16, n_fft=256,
                                         hop_length=128, cwt_n_scales=8)
        vib = compute_vibration_input_stack(accel, sample_rate=376.0)
        np.testing.assert_array_equal(ac, ac_ref)
        np.testing.assert_array_equal(vib, vib_ref)


def test_cluster_metric_flags_collapse() -> None:
    # All-identical embeddings → K-means finds one populated cluster → NMI 0.
    emb = np.ones((30, 4), dtype=np.float64)
    labels = (["Pump"] * 10) + (["Turbine"] * 10) + (["Standstill"] * 10)
    m = cluster_purity_and_nmi(emb, labels, n_clusters=3, seed=0)
    assert m["nmi"] == 0.0
    assert m["collapsed"] is True
    assert m["n_effective_clusters"] == 1

    # A genuinely separated embedding must NOT be flagged as collapsed.
    rng = np.random.default_rng(0)
    sep = np.concatenate([
        rng.normal(loc, 0.05, size=(10, 4)) for loc in (-5.0, 0.0, 5.0)
    ])
    m2 = cluster_purity_and_nmi(sep, labels, n_clusters=3, seed=0)
    assert m2["collapsed"] is False
    assert m2["nmi"] > 0.5
