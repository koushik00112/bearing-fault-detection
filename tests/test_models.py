import numpy as np
import pytest

from bearing import models
from bearing.splits import build_splits


@pytest.fixture(scope="module")
def split_sets(ws):
    [s] = build_splits(ws.meta, "B_recording", seed=0)
    return ws.subset(s.train), ws.subset(s.test)


@pytest.mark.parametrize("name", ["rf", "hgb"])
def test_baselines_fit_predict_and_round_trip(split_sets, tmp_path, name):
    train, test = split_sets
    m = models.make(name, ("healthy", "inner", "outer"), train.fs, n_estimators=20, max_iter=20)
    m.fit(train, None, seed=0)
    p = m.predict_proba(test.X)
    assert p.shape == (len(test), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    m.save(tmp_path)
    np.testing.assert_allclose(models.load(tmp_path).predict_proba(test.X[:20]), p[:20])


def test_missing_training_class_gets_zero_probability(ws):
    only_two = ws.subset(np.flatnonzero(ws.meta["label"] != "inner"))
    m = models.make("rf", ("healthy", "inner", "outer"), ws.fs, n_estimators=10)
    m.fit(only_two, None, seed=0)
    assert np.all(m.predict_proba(ws.X[:10])[:, 1] == 0)


@pytest.mark.cnn
def test_cnn_trains_with_early_stopping_and_round_trips(split_sets, tmp_path):
    train, test = split_sets
    m = models.make("cnn", ("healthy", "inner", "outer"), train.fs, epochs=2, device="cpu")
    m.fit(train, test.subset(np.arange(30)), seed=0)
    assert len(m.history) == 2 and "val_loss" in m.history[0]
    p = m.predict_proba(test.X)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
    m.save(tmp_path)
    np.testing.assert_allclose(models.load(tmp_path).predict_proba(test.X[:20]), p[:20], atol=1e-5)
    assert m.n_parameters() < 100_000


def test_unknown_model():
    with pytest.raises(ValueError):
        models.make("svm", ("a",), 1.0)
