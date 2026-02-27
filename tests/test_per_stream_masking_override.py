# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for per-stream masking override infrastructure."""

from omegaconf import OmegaConf

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


def _make_mode_cfg():
    """Create a minimal mode_cfg with model_input and target_input."""
    return OmegaConf.create(
        {
            "model_input": {
                "random_easy": {
                    "masking_strategy": "random",
                    "num_samples": 1,
                    "num_steps_input": 1,
                    "masking_strategy_config": {
                        "rate": 0.6,
                        "rate_sampling": False,
                    },
                },
            },
            "target_input": {
                "healpix_target": {
                    "masking_strategy": "healpix",
                    "num_samples": 1,
                    "masking_strategy_config": {
                        "rate": 0.2,
                        "hl_mask": 0,
                        "rate_sampling": False,
                    },
                },
            },
            "training_mode": ["student_teacher"],
            "losses": {},
            "samples_per_mini_epoch": 128,
            "shuffle": True,
        }
    )


def _make_stream_info(name, masking_override=None):
    """Create a minimal stream_info dict."""
    info = OmegaConf.create(
        {
            "name": name,
            "type": "anemoi",
            "stream_id": 0,
            "filenames": [],
        }
    )
    if masking_override is not None:
        info["masking_override"] = masking_override
    return info


class TestBuildEffectiveMaskingCfgs:
    """Test _build_effective_masking_cfgs logic in isolation."""

    def _build(self, streams, mode_cfg):
        """Call _build_effective_masking_cfgs without constructing a full sampler.

        We monkey-patch the minimal attributes the method needs.
        """
        obj = object.__new__(MultiStreamDataSampler)
        obj.streams = streams
        obj.mode_cfg = mode_cfg
        return obj._build_effective_masking_cfgs()

    def test_no_override_returns_global_config(self):
        mode_cfg = _make_mode_cfg()
        streams = [_make_stream_info("ERA5")]
        cfgs = self._build(streams, mode_cfg)

        assert "ERA5" in cfgs
        # Should be the exact same object (no copy)
        assert cfgs["ERA5"] is mode_cfg

    def test_override_masking_strategy_config(self):
        mode_cfg = _make_mode_cfg()
        override = OmegaConf.create(
            {
                "target_input": {
                    "masking_strategy_config": {"hl_mask": 2},
                },
            }
        )
        streams = [_make_stream_info("ERA5_z", masking_override=override)]
        cfgs = self._build(streams, mode_cfg)

        effective = cfgs["ERA5_z"]
        # Should NOT be the same object (deep copied)
        assert effective is not mode_cfg

        # target_input should have overridden hl_mask
        target_cfg = list(effective.target_input.values())[0]
        assert target_cfg.masking_strategy_config.hl_mask == 2
        # Non-overridden keys should be preserved
        assert target_cfg.masking_strategy_config.rate == 0.2
        assert target_cfg.masking_strategy_config.rate_sampling is False

        # model_input should be unchanged
        model_cfg = list(effective.model_input.values())[0]
        assert model_cfg.masking_strategy_config.rate == 0.6

    def test_override_masking_strategy(self):
        mode_cfg = _make_mode_cfg()
        override = OmegaConf.create(
            {
                "model_input": {
                    "masking_strategy": "healpix",
                    "masking_strategy_config": {"hl_mask": 3, "rate": 0.5},
                },
            }
        )
        streams = [_make_stream_info("ERA5_precip", masking_override=override)]
        cfgs = self._build(streams, mode_cfg)

        effective = cfgs["ERA5_precip"]
        model_cfg = list(effective.model_input.values())[0]
        assert model_cfg.masking_strategy == "healpix"
        assert model_cfg.masking_strategy_config.hl_mask == 3
        assert model_cfg.masking_strategy_config.rate == 0.5

    def test_partial_override_preserves_non_overridden_keys(self):
        mode_cfg = _make_mode_cfg()
        # Only override hl_mask in target, leave everything else
        override = OmegaConf.create(
            {
                "target_input": {
                    "masking_strategy_config": {"hl_mask": 1},
                },
            }
        )
        streams = [_make_stream_info("stream_a", masking_override=override)]
        cfgs = self._build(streams, mode_cfg)

        effective = cfgs["stream_a"]
        target_cfg = list(effective.target_input.values())[0]
        # Overridden
        assert target_cfg.masking_strategy_config.hl_mask == 1
        # Preserved from original
        assert target_cfg.masking_strategy_config.rate == 0.2
        assert target_cfg.masking_strategy == "healpix"
        # model_input untouched
        model_cfg = list(effective.model_input.values())[0]
        assert model_cfg.masking_strategy == "random"
        assert model_cfg.masking_strategy_config.rate == 0.6

    def test_multiple_streams_different_overrides(self):
        mode_cfg = _make_mode_cfg()
        override_a = OmegaConf.create(
            {
                "target_input": {
                    "masking_strategy_config": {"hl_mask": 1},
                },
            }
        )
        override_b = OmegaConf.create(
            {
                "target_input": {
                    "masking_strategy_config": {"hl_mask": 3},
                },
            }
        )
        streams = [
            _make_stream_info("stream_a", masking_override=override_a),
            _make_stream_info("stream_b", masking_override=override_b),
            _make_stream_info("stream_c"),  # no override
        ]
        cfgs = self._build(streams, mode_cfg)

        # Each stream should have independent config
        cfg_a = list(cfgs["stream_a"].target_input.values())[0]
        cfg_b = list(cfgs["stream_b"].target_input.values())[0]

        assert cfg_a.masking_strategy_config.hl_mask == 1
        assert cfg_b.masking_strategy_config.hl_mask == 3
        # stream_c uses global
        assert cfgs["stream_c"] is mode_cfg

    def test_override_does_not_mutate_global_config(self):
        mode_cfg = _make_mode_cfg()
        original_hl_mask = list(mode_cfg.target_input.values())[0].masking_strategy_config.hl_mask

        override = OmegaConf.create(
            {
                "target_input": {
                    "masking_strategy_config": {"hl_mask": 4},
                },
            }
        )
        streams = [_make_stream_info("stream_a", masking_override=override)]
        self._build(streams, mode_cfg)

        # Global config should not be mutated
        assert (
            list(mode_cfg.target_input.values())[0].masking_strategy_config.hl_mask
            == original_hl_mask
        )

    def test_empty_override_returns_copy(self):
        mode_cfg = _make_mode_cfg()
        override = OmegaConf.create({})
        streams = [_make_stream_info("stream_a", masking_override=override)]
        cfgs = self._build(streams, mode_cfg)

        # With an empty override, we still deep-copy but values match
        effective = cfgs["stream_a"]
        assert effective is not mode_cfg
        target_cfg = list(effective.target_input.values())[0]
        assert target_cfg.masking_strategy_config.hl_mask == 0
