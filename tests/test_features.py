import numpy as np

from bearing import features


def test_shape_names_and_finiteness(ws):
    F = features.extract(ws.X[:50], ws.fs)
    names = features.feature_names(ws.channels)
    assert F.shape == (50, len(names)) == (50, 3 * features.FEATURES_PER_CHANNEL)
    assert np.isfinite(F).all()


def test_known_values_for_a_pure_sine():
    fs, n = 16_000.0, 4096
    t = np.arange(n) / fs
    x = (2.0 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[None, None, :]
    F = features.extract(x, fs)[0]
    names = features.feature_names(("s",))
    f = dict(zip(names, F, strict=True))
    assert abs(f["s.rms"] - 2 / np.sqrt(2)) < 1e-2
    assert abs(f["s.crest"] - np.sqrt(2)) < 1e-2
    assert abs(f["s.centroid"] - 1000) < 50
    assert f["s.kurtosis"] < -1  # a sine is platykurtic (excess kurtosis -1.5)


def test_impulses_raise_kurtosis():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1, 1, 4096)).astype(np.float32)
    spiky = noise.copy()
    spiky[0, 0, ::400] += 20
    k_noise = features.extract(noise, 16_000)[0][4]
    k_spiky = features.extract(spiky, 16_000)[0][4]
    assert k_spiky > k_noise + 5


def test_empty_input():
    assert features.extract(np.empty((0, 2, 256), np.float32), 1000).shape == (
        0,
        2 * features.FEATURES_PER_CHANNEL,
    )
