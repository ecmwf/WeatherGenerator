import numpy as np

from weathergen.datasets.masking import Masker


def test_sampling_rate_without_schedule_stays_fixed():
    masker = Masker(healpix_level=0, stage="train")
    masker.reset_rng(np.random.default_rng(0), mini_epoch=3)

    rate = masker._get_sampling_rate({"rate": 0.1})

    assert rate == 0.1


def test_sampling_rate_linearly_interpolates_over_mini_epochs():
    masker = Masker(
        healpix_level=0,
        stage="train",
        mode_cfg={"num_mini_epochs": 5},
    )

    masker.reset_rng(np.random.default_rng(0), mini_epoch=0)
    assert masker._get_sampling_rate({"rate": 0.1, "rate_end": 0.5}) == 0.1

    masker.reset_rng(np.random.default_rng(0), mini_epoch=2)
    assert masker._get_sampling_rate({"rate": 0.1, "rate_end": 0.5}) == 0.3

    masker.reset_rng(np.random.default_rng(0), mini_epoch=4)
    assert masker._get_sampling_rate({"rate": 0.1, "rate_end": 0.5}) == 0.5