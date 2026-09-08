# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for checkpoint-key ``module.`` prefix alignment.

``_align_module_prefix`` and ``_strip_module_prefix`` are pure dict/str functions, so they are
exercised here without torch.distributed. They matter because both load paths resolve parameters
by exact name: ``load_model``'s sharded branch and ``load_decoder_from_checkpoint``'s sharded
branch each look names up in ``model.state_dict()``, and a convention mismatch makes every lookup
miss. In ``load_decoder_from_checkpoint`` that used to happen silently (each miss was skipped
individually and an empty load still "succeeded"), so a DDP-saved backbone would train a randomly
initialised decoder. The sharded branches themselves need real FSDP and are covered by the
integration check in the skill notes, not here.
"""

import pytest

from weathergen.model.model_interface import (
    _DECODER_PREFIXES,
    _align_module_prefix,
    _strip_module_prefix,
)

# Shapes of the real key conventions: a DDP-wrapped save vs a bare model state_dict.
_PREFIXED = ["module.encoder.q_cells", "module.embed_target_coords.ERA5.linear.weight"]
_BARE = ["encoder.q_cells", "embed_target_coords.ERA5.linear.weight"]


def _sd(keys):
    return {k: object() for k in keys}


# --------------------------------------------------------------------------------------
# _strip_module_prefix
# --------------------------------------------------------------------------------------


def test_strip_removes_only_a_leading_prefix():
    assert _strip_module_prefix("module.encoder.weight") == "encoder.weight"
    assert _strip_module_prefix("encoder.weight") == "encoder.weight"


def test_strip_does_not_touch_module_elsewhere_in_the_path():
    """Regression: the old code used key.replace("module.", ""), which corrupts any path with
    a genuine submodule named "module" in it."""
    assert _strip_module_prefix("module.encoder.module.weight") == "encoder.module.weight"
    assert _strip_module_prefix("encoder.module.weight") == "encoder.module.weight"


# --------------------------------------------------------------------------------------
# _align_module_prefix
# --------------------------------------------------------------------------------------


def test_adds_prefix_when_model_has_one_and_params_do_not():
    out = _align_module_prefix(_sd(_BARE), _sd(_PREFIXED))
    assert list(out) == _PREFIXED


def test_strips_prefix_when_params_have_one_and_model_does_not():
    """The case that bit every dcft off a DDP-saved backbone (cw6a4szu, nhv6tkln)."""
    out = _align_module_prefix(_sd(_PREFIXED), _sd(_BARE))
    assert list(out) == _BARE


@pytest.mark.parametrize("keys", [_BARE, _PREFIXED])
def test_matching_conventions_are_left_alone(keys):
    params = _sd(keys)
    out = _align_module_prefix(params, _sd(keys))
    assert list(out) == keys
    assert all(out[k] is params[k] for k in keys)


def test_empty_params_are_returned_unchanged():
    assert _align_module_prefix({}, _sd(_PREFIXED)) == {}


def test_values_are_preserved_across_realignment():
    params = _sd(_PREFIXED)
    out = _align_module_prefix(params, _sd(_BARE))
    for src, dst in zip(_PREFIXED, _BARE, strict=True):
        assert out[dst] is params[src]


def test_alignment_is_an_involution_between_the_two_conventions():
    """Round-tripping must land back on the original keys -- the guard against a fix that
    half-strips or double-prefixes."""
    bare = _sd(_BARE)
    there = _align_module_prefix(bare, _sd(_PREFIXED))
    back = _align_module_prefix(there, _sd(_BARE))
    assert list(back) == _BARE


# --------------------------------------------------------------------------------------
# the decoder filter must see through either convention
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "is_decoder"),
    [
        ("embed_target_coords.ERA5.linear.weight", True),
        ("module.embed_target_coords.ERA5.linear.weight", True),
        ("target_token_engines.ERA5.tte.0.lnorm_in_q.embed_aux.0.weight", True),
        ("module.target_token_engines.ERA5.tte.0.lnorm_in_q.embed_aux.0.weight", True),
        ("pred_heads.ERA5.0.weight", True),
        ("module.pred_heads.ERA5.0.weight", True),
        ("encoder.q_cells", False),
        ("module.encoder.q_cells", False),
        ("forecast_engine.net.fe_blocks.0.layers.0.weight", False),
        ("module.forecast_engine.net.fe_blocks.0.layers.0.weight", False),
    ],
)
def test_decoder_prefix_filter_matches_under_both_conventions(key, is_decoder):
    assert _strip_module_prefix(key).startswith(_DECODER_PREFIXES) is is_decoder
