import pytest

from bearing.data import synthetic
from bearing.data.paderborn import CHANNELS
from bearing.windowing import make_windows


@pytest.fixture(scope="session")
def recordings():
    # 3 classes x 3 bearings x 4 conditions x 3 reps = 108 recordings, 0.5 s each.
    return synthetic.make_dataset(bearings_per_class=3, reps=3, seconds=0.5, seed=1)


@pytest.fixture(scope="session")
def ws(recordings):
    return make_windows(recordings, CHANNELS, window=2048, stride=1024)
