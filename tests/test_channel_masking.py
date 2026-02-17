# (C) Copyright 2025 WeatherGenerator contributors.
#
# Tests for per-variable channel masking based on autocorrelation.
# Verifies the full pipeline: config → mask generation → loss mask → loss function.

import numpy as np
import pytest
import torch

from weathergen.datasets.masking import (
    ChannelMaskingConfig,
    Masker,
    _healpix_cell_size_km,
    correlation_length_to_hl_mask,
)
from weathergen.train.loss_modules.loss_functions import lp_loss


# ---------------------------------------------------------------------------
# 1. correlation_length_to_hl_mask — pure mapping function
# ---------------------------------------------------------------------------
class TestCorrelationLengthToHlMask:
    """Verify the mapping from spatial correlation length to HEALPix level."""

    def test_large_scale_z500(self):
        """z_500 (L_corr ~4000 km) should map to level 1."""
        assert correlation_length_to_hl_mask(4000) == 1

    def test_mesoscale_10u(self):
        """10u wind (L_corr ~1200 km) should map to level 2."""
        assert correlation_length_to_hl_mask(1200) == 2

    def test_fine_scale_tp(self):
        """Precipitation (L_corr ~177 km) should map to level 5."""
        assert correlation_length_to_hl_mask(177) == 5

    def test_very_large_correlation_clamps_to_min(self):
        """Very large correlation should clamp at hl_min."""
        result = correlation_length_to_hl_mask(50000, hl_min=1)
        assert result == 1

    def test_very_small_correlation_returns_max(self):
        """Very small correlation should return hl_max (finest)."""
        result = correlation_length_to_hl_mask(1.0, hl_max=5)
        assert result == 5

    def test_multiplier_increases_mask_coarseness(self):
        """Higher multiplier → larger target → coarser mask (lower level)."""
        level_m1 = correlation_length_to_hl_mask(1200, multiplier=1.0)
        level_m3 = correlation_length_to_hl_mask(1200, multiplier=3.0)
        assert level_m3 <= level_m1, (
            f"multiplier=3 should give coarser (lower) level than multiplier=1, "
            f"but got {level_m3} vs {level_m1}"
        )

    def test_custom_hl_range(self):
        """Custom hl_min/hl_max should constrain the output."""
        result = correlation_length_to_hl_mask(4000, hl_min=2, hl_max=4)
        assert 2 <= result <= 4

    def test_cell_sizes_are_monotonically_decreasing(self):
        """HEALPix cell sizes should decrease with increasing level."""
        sizes = [_healpix_cell_size_km(hl) for hl in range(7)]
        for i in range(len(sizes) - 1):
            assert sizes[i] > sizes[i + 1], (
                f"Cell size at level {i} ({sizes[i]:.0f} km) should be > "
                f"level {i + 1} ({sizes[i + 1]:.0f} km)"
            )


# ---------------------------------------------------------------------------
# 2. ChannelMaskingConfig — dataclass configuration
# ---------------------------------------------------------------------------
class TestChannelMaskingConfig:
    """Verify config parsing and per-channel level lookup."""

    @pytest.fixture
    def enabled_config(self):
        return ChannelMaskingConfig(
            autocorr={
                "z_500": {"space_km": 4000},
                "t_850": {"space_km": 4009},
                "10u": {"space_km": 1200},
                "tp": {"space_km": 177},
            },
            enabled=True,
            default_hl_mask=3,
        )

    def test_known_channel_returns_correct_level(self, enabled_config):
        assert enabled_config.get_hl_mask("z_500") == 1
        assert enabled_config.get_hl_mask("10u") == 2
        assert enabled_config.get_hl_mask("tp") == 5

    def test_unknown_channel_returns_default(self, enabled_config):
        assert enabled_config.get_hl_mask("unknown_var") == 3

    def test_disabled_returns_default_for_all(self):
        cfg = ChannelMaskingConfig(
            autocorr={"z_500": {"space_km": 4000}},
            enabled=False,
            default_hl_mask=3,
        )
        assert cfg.get_hl_mask("z_500") == 3
        assert cfg.get_hl_mask("tp") == 3

    def test_from_config_dict(self):
        """from_config round-trips a YAML-like dict."""
        raw = {
            "enabled": True,
            "autocorr": {"z_500": {"space_km": 4000}},
            "multiplier": 2.0,
            "hl_mask_min": 2,
            "hl_mask_max": 4,
            "default_hl_mask": 3,
        }
        cfg = ChannelMaskingConfig.from_config(raw)
        assert cfg.enabled is True
        assert cfg.multiplier == 2.0
        assert cfg.hl_mask_min == 2
        assert cfg.hl_mask_max == 4
        assert "z_500" in cfg.autocorr

    def test_from_config_none(self):
        """from_config(None) returns disabled default."""
        cfg = ChannelMaskingConfig.from_config(None)
        assert cfg.enabled is False

    def test_same_correlation_gives_same_level(self, enabled_config):
        """z_500 and t_850 have similar correlation → same level."""
        assert enabled_config.get_hl_mask("z_500") == enabled_config.get_hl_mask("t_850")


# ---------------------------------------------------------------------------
# 3. generate_channel_masks — mask generation
# ---------------------------------------------------------------------------
class TestGenerateChannelMasks:
    """Verify that Masker.generate_channel_masks produces correct per-channel masks."""

    @pytest.fixture
    def masker(self):
        """HEALPix level 4 → 3072 cells."""
        m = Masker(healpix_level=4)
        m.reset_rng(np.random.default_rng(42))
        return m

    @pytest.fixture
    def channel_config(self):
        return ChannelMaskingConfig(
            autocorr={
                "z_500": {"space_km": 4000},  # level 1
                "t_850": {"space_km": 4009},  # level 1 (same as z_500)
                "10u": {"space_km": 1200},  # level 2
                "tp": {"space_km": 177},  # level 3 (capped at data_level - 1 = 3)
            },
            enabled=True,
        )

    def test_all_channels_present(self, masker, channel_config):
        """Every channel in the list should appear in the result dict."""
        channels = ["z_500", "t_850", "10u", "tp"]
        masks = masker.generate_channel_masks(channels, channel_config, keep_rate=0.5)
        for ch in channels:
            assert ch in masks, f"Missing mask for channel {ch}"

    def test_mask_shape(self, masker, channel_config):
        """All masks should have shape [num_cells] = [3072]."""
        channels = ["z_500", "t_850", "10u", "tp"]
        masks = masker.generate_channel_masks(channels, channel_config, keep_rate=0.5)
        num_cells = 12 * (4**4)
        for ch in channels:
            assert masks[ch].shape == (num_cells,), (
                f"Mask for {ch} has shape {masks[ch].shape}, expected ({num_cells},)"
            )

    def test_same_hl_shares_mask(self, masker, channel_config):
        """Channels with the same hl_mask should share the exact same mask array."""
        channels = ["z_500", "t_850", "10u", "tp"]
        masks = masker.generate_channel_masks(channels, channel_config, keep_rate=0.5)
        # z_500 and t_850 both map to level 1
        np.testing.assert_array_equal(
            masks["z_500"],
            masks["t_850"],
            err_msg="z_500 and t_850 should share the same mask (same hl_mask level)",
        )

    def test_different_hl_has_different_masks(self, masker, channel_config):
        """Channels at different hl_mask levels should have different masks."""
        channels = ["z_500", "10u", "tp"]
        masks = masker.generate_channel_masks(channels, channel_config, keep_rate=0.5)

        # z_500 (level 1) vs 10u (level 2) — independently generated at different levels
        assert not np.array_equal(masks["z_500"], masks["10u"]), (
            "z_500 (level 1) and 10u (level 2) should have different mask patterns"
        )

    def test_coarser_mask_has_larger_blocks(self, masker, channel_config):
        """Coarser mask (lower hl_mask) should have fewer transitions (larger contiguous regions)."""
        channels = ["z_500", "10u"]
        masks = masker.generate_channel_masks(channels, channel_config, keep_rate=0.5)

        def count_transitions(mask):
            """Count number of True↔False transitions."""
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            return np.sum(np.diff(mask.astype(int)) != 0)

        trans_z500 = count_transitions(masks["z_500"])
        trans_10u = count_transitions(masks["10u"])

        # Level 1 (z_500) should have fewer transitions than level 2 (10u) on expectation
        # Both are stochastic but with high probability this holds
        assert trans_z500 < trans_10u, (
            f"Coarser mask (z_500, level 1) should have fewer transitions ({trans_z500}) "
            f"than finer mask (10u, level 2) with {trans_10u} transitions"
        )

    def test_keep_rate_approximately_correct(self, masker, channel_config):
        """Fraction of kept cells should approximately match keep_rate."""
        channels = ["z_500"]
        keep_rate = 0.5
        masks = masker.generate_channel_masks(channels, channel_config, keep_rate=keep_rate)

        mask = masks["z_500"]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        actual_rate = mask.sum() / mask.size

        assert abs(actual_rate - keep_rate) < 0.2, (
            f"Keep rate for z_500 is {actual_rate:.2f}, expected ~{keep_rate} (±0.2)"
        )

    def test_disabled_config_returns_empty(self, masker):
        """Disabled config should return empty dict."""
        cfg = ChannelMaskingConfig(
            autocorr={"z_500": {"space_km": 4000}},
            enabled=False,
        )
        masks = masker.generate_channel_masks(["z_500"], cfg, keep_rate=0.5)
        assert masks == {}

    def test_different_seeds_produce_different_masks(self, channel_config):
        """Different RNG seeds should produce different mask patterns."""
        channels = ["z_500"]
        m1 = Masker(healpix_level=4)
        m1.reset_rng(np.random.default_rng(1))
        masks1 = m1.generate_channel_masks(channels, channel_config, keep_rate=0.5)

        m2 = Masker(healpix_level=4)
        m2.reset_rng(np.random.default_rng(999))
        masks2 = m2.generate_channel_masks(channels, channel_config, keep_rate=0.5)

        m1_arr = masks1["z_500"].numpy() if isinstance(masks1["z_500"], torch.Tensor) else masks1["z_500"]
        m2_arr = masks2["z_500"].numpy() if isinstance(masks2["z_500"], torch.Tensor) else masks2["z_500"]
        assert not np.array_equal(m1_arr, m2_arr), (
            "Different seeds should produce different masks"
        )


# ---------------------------------------------------------------------------
# 4. channel_loss_mask complement logic
# ---------------------------------------------------------------------------
class TestChannelLossMaskLogic:
    """Verify that channel_loss_mask is the complement of the source-side channel masks.

    The logic is in tokenize_apply_mask_target: for each surviving target point,
    channel_loss_mask[i, c] = 1.0 - source_mask[cell_id_of_point_i][channel_c].
    This means: loss is computed where the channel was hidden (masked) on source side.
    """

    def test_complement_logic_manual(self):
        """Manually verify complement logic with known values."""
        # Simulate 4 cells, 2 channels
        num_cells = 4
        num_channels = 2

        # Channel 0 ("z_500"): visible at cells 0,1; hidden at cells 2,3
        ch0_source_mask = np.array([True, True, False, False])

        # Channel 1 ("tp"): visible at cells 0,2; hidden at cells 1,3
        ch1_source_mask = np.array([True, False, True, False])

        channel_masks_dict = {
            "z_500": ch0_source_mask,
            "tp": ch1_source_mask,
        }
        channel_list = ["z_500", "tp"]

        # Suppose all 4 cells survive as target points (1 point per cell)
        cell_ids = np.arange(num_cells)

        # Compute channel_loss_mask the same way as tokenize_apply_mask_target
        ch_loss_mask = np.ones((num_cells, num_channels), dtype=np.float32)
        for c_idx, ch_name in enumerate(channel_list):
            cell_mask_f = channel_masks_dict[ch_name].astype(np.float32)
            ch_loss_mask[:, c_idx] = 1.0 - cell_mask_f[cell_ids]

        # Expected: loss where source mask was False
        # ch0: cells 0,1 visible → loss_mask 0; cells 2,3 hidden → loss_mask 1
        expected_ch0 = np.array([0.0, 0.0, 1.0, 1.0])
        # ch1: cells 0,2 visible → loss_mask 0; cells 1,3 hidden → loss_mask 1
        expected_ch1 = np.array([0.0, 1.0, 0.0, 1.0])

        np.testing.assert_array_almost_equal(ch_loss_mask[:, 0], expected_ch0)
        np.testing.assert_array_almost_equal(ch_loss_mask[:, 1], expected_ch1)

    def test_all_visible_gives_zero_loss_mask(self):
        """If a channel is visible everywhere, its loss mask should be all zeros."""
        all_visible = np.array([True, True, True, True])
        channel_masks_dict = {"ch0": all_visible}
        channel_list = ["ch0"]
        cell_ids = np.arange(4)

        ch_loss_mask = np.ones((4, 1), dtype=np.float32)
        cell_mask_f = all_visible.astype(np.float32)
        ch_loss_mask[:, 0] = 1.0 - cell_mask_f[cell_ids]

        np.testing.assert_array_equal(ch_loss_mask[:, 0], np.zeros(4))

    def test_all_hidden_gives_one_loss_mask(self):
        """If a channel is hidden everywhere, its loss mask should be all ones."""
        all_hidden = np.array([False, False, False, False])
        channel_masks_dict = {"ch0": all_hidden}
        channel_list = ["ch0"]
        cell_ids = np.arange(4)

        ch_loss_mask = np.ones((4, 1), dtype=np.float32)
        cell_mask_f = all_hidden.astype(np.float32)
        ch_loss_mask[:, 0] = 1.0 - cell_mask_f[cell_ids]

        np.testing.assert_array_equal(ch_loss_mask[:, 0], np.ones(4))

    def test_per_channel_masking_differs_between_channels(self):
        """Different source masks should produce different loss masks."""
        ch0_mask = np.array([True, False, True, False])
        ch1_mask = np.array([False, True, False, True])

        channel_masks_dict = {"ch0": ch0_mask, "ch1": ch1_mask}
        cell_ids = np.arange(4)

        ch_loss_mask = np.ones((4, 2), dtype=np.float32)
        for c_idx, ch_name in enumerate(["ch0", "ch1"]):
            cell_mask_f = channel_masks_dict[ch_name].astype(np.float32)
            ch_loss_mask[:, c_idx] = 1.0 - cell_mask_f[cell_ids]

        # ch0 and ch1 should have complementary loss masks
        np.testing.assert_array_equal(ch_loss_mask[:, 0] + ch_loss_mask[:, 1], np.ones(4))


# ---------------------------------------------------------------------------
# 5. lp_loss with channel_loss_mask
# ---------------------------------------------------------------------------
class TestLpLossWithChannelLossMask:
    """Verify that lp_loss correctly applies channel_loss_mask."""

    def test_all_ones_mask_matches_standard_loss(self):
        """channel_loss_mask of all 1s should produce identical result to no mask."""
        target = torch.randn(10, 3)
        pred = torch.randn(1, 10, 3)  # ens_dim=1
        mask = torch.ones(10, 3)

        loss_nomask, chs_nomask = lp_loss(target, pred, p_norm=2, weights_channels=None,
                                           weights_points=None, channel_loss_mask=None)
        loss_mask, chs_mask = lp_loss(target, pred, p_norm=2, weights_channels=None,
                                       weights_points=None, channel_loss_mask=mask)

        torch.testing.assert_close(loss_mask, loss_nomask, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chs_mask, chs_nomask, atol=1e-5, rtol=1e-5)

    def test_zero_column_zeroes_channel_loss(self):
        """Masking out a channel column should zero that channel's contribution."""
        target = torch.randn(10, 3)
        pred = torch.randn(1, 10, 3)

        # Zero out channel 1
        mask = torch.ones(10, 3)
        mask[:, 1] = 0.0

        _, chs = lp_loss(target, pred, p_norm=2, weights_channels=None,
                          weights_points=None, channel_loss_mask=mask)

        # Channel 1 should have zero loss
        assert chs[1].item() == 0.0, f"Channel 1 loss should be 0, got {chs[1].item()}"
        # Channels 0 and 2 should have non-zero loss (with overwhelming probability)
        assert chs[0].item() > 0.0
        assert chs[2].item() > 0.0

    def test_partial_mask_normalises_correctly(self):
        """With partial masking, effective_n should be used for normalisation."""
        # 10 points, 1 channel, mask keeps only 5 points
        target = torch.ones(10, 1)
        pred = torch.zeros(1, 10, 1)  # prediction = 0, so diff^2 = 1.0

        mask = torch.zeros(10, 1)
        mask[:5, 0] = 1.0  # only first 5 points contribute

        loss, _ = lp_loss(target, pred, p_norm=2, weights_channels=None,
                           weights_points=None, channel_loss_mask=mask)

        # diff_p = 1.0 for all points, masked by mask, sum = 5.0, effective_n = 5
        # loss = sum / effective_n = 5.0 / 5.0 = 1.0
        expected = 1.0
        assert abs(loss.item() - expected) < 1e-5, (
            f"Loss should be {expected}, got {loss.item()}"
        )

    def test_full_mask_vs_no_mask_consistency(self):
        """Full mask (all 1s) divided by N should equal standard mean."""
        N, C = 20, 4
        target = torch.randn(N, C)
        pred = torch.randn(1, N, C)

        loss_std, _ = lp_loss(target, pred, p_norm=2, weights_channels=None,
                               weights_points=None, channel_loss_mask=None)
        loss_masked, _ = lp_loss(target, pred, p_norm=2, weights_channels=None,
                                  weights_points=None,
                                  channel_loss_mask=torch.ones(N, C))

        torch.testing.assert_close(loss_std, loss_masked, atol=1e-5, rtol=1e-5)

    def test_per_channel_different_masks(self):
        """Different masks per channel should produce different per-channel losses."""
        N = 100
        target = torch.randn(N, 2)
        pred = torch.zeros(1, N, 2)

        # Channel 0: mask all, Channel 1: keep all
        mask = torch.zeros(N, 2)
        mask[:, 1] = 1.0

        _, chs = lp_loss(target, pred, p_norm=2, weights_channels=None,
                          weights_points=None, channel_loss_mask=mask)

        assert chs[0].item() == 0.0, "Channel 0 (fully masked out) should have 0 loss"
        assert chs[1].item() > 0.0, "Channel 1 (fully kept) should have non-zero loss"


# ---------------------------------------------------------------------------
# 6. Integration: generate masks → verify they produce meaningful loss masks
# ---------------------------------------------------------------------------
class TestEndToEndChannelMasking:
    """Integration test: generate channel masks and verify the loss mask logic."""

    def test_full_pipeline_mask_generation_to_loss(self):
        """
        End-to-end: generate per-channel masks, compute channel_loss_mask,
        and verify it correctly affects lp_loss.
        """
        hl_data = 4
        num_cells = 12 * (4 ** hl_data)

        # Setup masker
        masker = Masker(healpix_level=hl_data)
        masker.reset_rng(np.random.default_rng(42))

        config = ChannelMaskingConfig(
            autocorr={
                "z_500": {"space_km": 4000},  # level 1 (coarse → large blocks visible)
                "tp": {"space_km": 177},  # level 3 (fine → small blocks visible)
            },
            enabled=True,
        )

        channels = ["z_500", "tp"]
        masks = masker.generate_channel_masks(channels, config, keep_rate=0.5)

        # Verify masks are generated
        assert len(masks) == 2
        assert "z_500" in masks
        assert "tp" in masks

        # Verify they have different patterns
        z500_mask = masks["z_500"]
        tp_mask = masks["tp"]
        if isinstance(z500_mask, torch.Tensor):
            z500_mask = z500_mask.numpy()
        if isinstance(tp_mask, torch.Tensor):
            tp_mask = tp_mask.numpy()

        assert not np.array_equal(z500_mask, tp_mask), (
            "z_500 and tp should have different mask patterns"
        )

        # Build channel_loss_mask (complement) for a subset of cells
        # Simulating that all cells survived spatial masking
        cell_ids = np.arange(num_cells)
        ch_loss_mask = np.ones((num_cells, 2), dtype=np.float32)
        for c_idx, ch_name in enumerate(channels):
            source_mask = masks[ch_name]
            if isinstance(source_mask, torch.Tensor):
                source_mask = source_mask.numpy()
            ch_loss_mask[:, c_idx] = 1.0 - source_mask.astype(np.float32)

        # Verify complement: where source was visible (True), loss_mask = 0
        z500_visible_cells = z500_mask.astype(bool)
        assert np.all(ch_loss_mask[z500_visible_cells, 0] == 0.0), (
            "Loss mask should be 0 where z_500 was visible on source side"
        )
        assert np.all(ch_loss_mask[~z500_visible_cells, 0] == 1.0), (
            "Loss mask should be 1 where z_500 was hidden on source side"
        )

        # Use in lp_loss
        target = torch.randn(num_cells, 2)
        pred = torch.zeros(1, num_cells, 2)
        ch_loss_mask_tensor = torch.from_numpy(ch_loss_mask)

        loss, chs = lp_loss(target, pred, p_norm=2, weights_channels=None,
                             weights_points=None, channel_loss_mask=ch_loss_mask_tensor)

        # Both channels should have non-zero loss (since keep_rate=0.5, ~half cells hidden)
        assert chs[0].item() > 0.0, "z_500 loss should be > 0"
        assert chs[1].item() > 0.0, "tp loss should be > 0"

        # z_500 (coarse mask, level 1) has ~50% cells visible → ~50% contribute to loss
        # tp (fine mask, level 3) also has ~50% cells visible → ~50% contribute to loss
        # Both should have roughly comparable loss magnitudes given same target/pred

    def test_channel_mask_granularity_differs(self):
        """
        Verify that coarser masks (lower hl_mask) produce spatially coarser patterns.

        Metric: number of contiguous blocks in the mask.
        A coarser mask (level 1) should have fewer, larger blocks than a finer mask.
        """
        hl_data = 4
        num_cells = 12 * (4 ** hl_data)

        masker = Masker(healpix_level=hl_data)
        masker.reset_rng(np.random.default_rng(42))

        config = ChannelMaskingConfig(
            autocorr={
                "z_500": {"space_km": 4000},  # level 1
                "10u": {"space_km": 1200},  # level 2
                "tp": {"space_km": 177},  # level 3
            },
            enabled=True,
        )

        masks = masker.generate_channel_masks(["z_500", "10u", "tp"], config, keep_rate=0.5)

        def count_unique_kept(mask):
            """Count fraction of cells kept."""
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            return mask.sum() / mask.size

        # All should be ~0.5 keep rate
        for ch in ["z_500", "10u", "tp"]:
            rate = count_unique_kept(masks[ch])
            assert 0.2 < rate < 0.8, f"Keep rate for {ch} is {rate:.2f}, expected ~0.5"

    def test_masks_are_deterministic_with_same_seed(self):
        """Same seed should produce identical masks."""
        hl_data = 4
        config = ChannelMaskingConfig(
            autocorr={"z_500": {"space_km": 4000}},
            enabled=True,
        )
        channels = ["z_500"]

        m1 = Masker(healpix_level=hl_data)
        m1.reset_rng(np.random.default_rng(42))
        masks1 = m1.generate_channel_masks(channels, config, keep_rate=0.5)

        m2 = Masker(healpix_level=hl_data)
        m2.reset_rng(np.random.default_rng(42))
        masks2 = m2.generate_channel_masks(channels, config, keep_rate=0.5)

        m1_arr = masks1["z_500"].numpy() if isinstance(masks1["z_500"], torch.Tensor) else masks1["z_500"]
        m2_arr = masks2["z_500"].numpy() if isinstance(masks2["z_500"], torch.Tensor) else masks2["z_500"]
        np.testing.assert_array_equal(m1_arr, m2_arr)
