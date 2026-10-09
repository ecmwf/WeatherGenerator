"""
Standalone tests for the ERA5 grid caches (no GPU, no ERA5 files, no Slurm).

Covers tokenization, target coordinates and the anemoi_rt reader. Cached vs uncached results
are compared bitwise (`torch.equal` / `np.array_equal`). Heavy anemoi/earthkit imports are stubbed.

From this worktree:

    PYTHONPATH=src:packages/common/src:packages/readers_extra/src \\
      python -m pytest tests/test_grid_cache.py -q

    PYTHONPATH=src:packages/common/src:packages/readers_extra/src \\
      python tests/test_grid_cache.py
"""

import logging

import numpy as np
import pytest
import torch
from weathergen.common.io import IOReaderData

from weathergen.datasets.tokenizer_masking import (
    TokenizerMasking,
    _assert_identical,
    grid_cache_enabled,
    grid_cache_verify,
)
from weathergen.datasets.tokenizer_utils import (
    encode_times_target,
    target_coords_from_template,
    tokenize_apply_mask_target,
)

HL = 3
STREAM_INFO = {"stream_id": 1, "token_size": 32, "tokenize_spacetime": False}
NUM_GEOINFOS = 13  # as the ERA5 anemoi_rt stream
NUM_TIMES = 3  # time steps per window (6 for ERA5, fewer to keep the test fast)
STEP = np.timedelta64(6, "h")
T0 = np.datetime64("2020-01-01T00:00:00")


def _grid(num_lat=30, num_lon=60, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(
        np.linspace(-90.0, 90.0, num_lat), np.linspace(-180.0, 180.0, num_lon, endpoint=False)
    )
    latlon = np.stack([lat.ravel(), lon.ravel()], axis=1).astype(np.float32)
    # a few points exactly on cell borders / poles must be handled identically too
    latlon[:5] = rng.uniform(-90, 90, (5, 2)).astype(np.float32)
    return latlon


def _window(step, latlon, seed=0):
    """Window `step`: fixed grid tiled over NUM_TIMES hourly steps, new geoinfos and datetimes."""
    rng = np.random.default_rng(1000 * seed + step)
    n = latlon.shape[0]
    coords = np.vstack([latlon] * NUM_TIMES)
    start = T0 + step * STEP
    times = np.array([start + i * np.timedelta64(1, "h") for i in range(NUM_TIMES)])
    datetimes = np.repeat(times, n)
    geoinfos = rng.normal(size=(coords.shape[0], NUM_GEOINFOS))  # float64 as in the reader
    data = np.empty((0, 4))
    time_win = (start, start + NUM_TIMES * np.timedelta64(1, "h"))
    return IOReaderData(coords, geoinfos, data, datetimes), time_win


def _tokenizers():
    # go through the constructor (a previous name collision with grid_cache_verify crashed here)
    cached = TokenizerMasking(HL, None, grid_cache_option=True)
    plain = TokenizerMasking(HL, None, grid_cache_option=False)
    return cached, plain


def _run(tok, rdata, time_win, cell_mask):
    tokens = tok.get_tokens_windows(STREAM_INFO, [rdata], False)[0]
    return tokens, tok.get_target_coords(STREAM_INFO, rdata, tokens, time_win, cell_mask)


def _assert_same(res_a, res_b):
    names = ["datetimes", "coords", "coords_local", "coords_per_cell", "idxs_ord_inv"]
    for name, a, b in zip(names, res_a, res_b, strict=True):
        if a is None or b is None:
            assert a is None and b is None, name
        elif isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert torch.equal(a, b), name
        else:
            assert a.dtype == b.dtype and np.array_equal(a, b), name


def _tokens_same(tok_a, tok_b):
    (ia, la), (ib, lb) = tok_a, tok_b
    assert len(ia) == len(ib) and la == lb
    for ca, cb in zip(ia, ib, strict=True):
        assert len(ca) == len(cb)
        for ta, tb in zip(ca, cb, strict=True):
            assert torch.equal(ta, tb)


@pytest.mark.parametrize("mask_kind", ["all", "partial"])
def test_cached_equals_uncached_over_steps(mask_kind):
    latlon = _grid()
    num_cells = 12 * 4**HL
    if mask_kind == "all":
        cell_mask = np.ones(num_cells, dtype=bool)
    else:
        cell_mask = np.random.default_rng(3).random(num_cells) > 0.4

    cached, plain = _tokenizers()
    previous_tokens = None
    for step in range(8):
        rdata_c, time_win = _window(step, latlon)
        rdata_p, _ = _window(step, latlon)
        tok_c, res_c = _run(cached, rdata_c, time_win, cell_mask)
        tok_p, res_p = _run(plain, rdata_p, time_win, cell_mask)

        _tokens_same(tok_c, tok_p)
        _assert_same(res_c, res_p)
        # tokenization is reused from the 2nd window on, outputs from the 3rd window on
        if previous_tokens is not None:
            assert tok_c[0] is previous_tokens[0]
        previous_tokens = tok_c

    entry = cached._target_coords_cache[STREAM_INFO["stream_id"]]
    assert entry.coords_local is not None, "target coords cache was never filled"
    assert not plain._tokens_cache and not plain._target_coords_cache


def test_outputs_do_not_alias_cache():
    """Modifying returned tensors in place must not change later results."""
    latlon = _grid()
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    cached, plain = _tokenizers()
    for step in range(6):
        rdata_c, time_win = _window(step, latlon)
        rdata_p, _ = _window(step, latlon)
        _, res_c = _run(cached, rdata_c, time_win, cell_mask)
        _, res_p = _run(plain, rdata_p, time_win, cell_mask)
        _assert_same(res_c, res_p)
        # vandalise everything that was handed out
        for t in res_c[1:]:
            t.mul_(-3)
            t.add_(7)


def test_changed_coords_invalidate_cache():
    latlon = _grid()
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    cached, plain = _tokenizers()

    def go(step, ll):
        rdata_c, time_win = _window(step, ll)
        rdata_p, _ = _window(step, ll)
        tok_c, res_c = _run(cached, rdata_c, time_win, cell_mask)
        tok_p, res_p = _run(plain, rdata_p, time_win, cell_mask)
        _tokens_same(tok_c, tok_p)
        _assert_same(res_c, res_p)

    for step in range(4):  # fill caches
        go(step, latlon)

    moved = latlon.copy()
    moved[100] += np.float32(0.5)  # one point moves (may or may not change its cell)
    go(4, moved)
    go(5, moved)
    go(6, moved)

    shuffled = latlon[np.random.default_rng(5).permutation(latlon.shape[0])]
    for step in range(7, 11):
        go(step, shuffled)

    fewer = latlon[:-17]
    for step in range(11, 14):
        go(step, fewer)

    for step in range(14, 17):  # back to the original grid
        go(step, latlon)


def test_changed_mask_invalidates_cache():
    latlon = _grid()
    num_cells = 12 * 4**HL
    cached, plain = _tokenizers()
    masks = [np.ones(num_cells, dtype=bool)] * 3
    rng = np.random.default_rng(11)
    masks += [rng.random(num_cells) > 0.5] * 3
    masks += [rng.random(num_cells) > 0.2] * 2
    masks += [np.ones(num_cells, dtype=bool)] * 2
    for step, cell_mask in enumerate(masks):
        rdata_c, time_win = _window(step, latlon)
        rdata_p, _ = _window(step, latlon)
        _, res_c = _run(cached, rdata_c, time_win, cell_mask)
        _, res_p = _run(plain, rdata_p, time_win, cell_mask)
        _assert_same(res_c, res_p)


def test_all_masked_out_and_empty_windows():
    latlon = _grid()
    num_cells = 12 * 4**HL
    cached, plain = _tokenizers()
    none_mask = np.zeros(num_cells, dtype=bool)
    all_mask = np.ones(num_cells, dtype=bool)
    for step, cell_mask in enumerate(
        [all_mask, all_mask, all_mask, none_mask, none_mask, all_mask]
    ):
        rdata_c, time_win = _window(step, latlon)
        rdata_p, _ = _window(step, latlon)
        _, res_c = _run(cached, rdata_c, time_win, cell_mask)
        _, res_p = _run(plain, rdata_p, time_win, cell_mask)
        _assert_same(res_c, res_p)

    empty = IOReaderData(
        np.zeros((0, 2), np.float32),
        np.zeros((0, NUM_GEOINFOS)),
        np.zeros((0, 4)),
        np.zeros((0,), dtype="datetime64[ns]"),
    )
    assert cached.get_tokens_windows(STREAM_INFO, [empty], False) == [(None, None)]


def test_spacetime_streams_are_not_cached():
    latlon = _grid()
    cached, _ = _tokenizers()
    info = dict(STREAM_INFO, tokenize_spacetime=True)
    rdata, _ = _window(0, latlon)
    rdata.data = np.zeros((rdata.coords.shape[0], 4))  # spacetime tokenization slices data rows
    cached.get_tokens_windows(info, [rdata], False)
    assert not cached._tokens_cache


def test_target_coords_from_template_matches_direct_computation():
    """Layout assumption: time columns [1, 1+T), geoinfo columns [1+T, 1+T+G), rest geometry."""
    tok = TokenizerMasking(HL, None)
    latlon = _grid()
    rdata, time_win = _window(0, latlon)
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    tokens, _ = _run(tok, rdata, time_win, cell_mask)
    mask_tokens, _ = tok.cell_to_token_mask(tokens[0], tokens[1], cell_mask)

    def direct(rd, tw):
        _, _, _, cl, _ = tokenize_apply_mask_target(
            1, tok.hl_target, tokens[0], tokens[1], mask_tokens, None, rd, tw,
            tok.hpy_verts_rots_target, tok.hpy_verts_local_target, tok.hpy_nctrs_target,
            encode_times_target,
        )  # fmt: skip
        return cl

    template = direct(rdata, time_win)
    for step in range(1, 4):
        rd, tw = _window(step, latlon)
        rd.coords = torch.tensor(rd.coords)
        rd.geoinfos = torch.tensor(rd.geoinfos)
        rd.data = torch.tensor(rd.data)
        want = direct(rd, tw)
        got = target_coords_from_template(
            template,
            rd.geoinfos[torch.cat([i for c in tokens[0] for i in c])],
            encode_times_target(rd.datetimes[torch.cat([i for c in tokens[0] for i in c])], tw),
        )
        assert torch.equal(want, got)


# --------------------------------------------------------------------------------------------
# anemoi_rt reader
# --------------------------------------------------------------------------------------------


def _import_reader_module(monkeypatch):
    """Import data_reader_anemoi_rt; stub the heavy anemoi/earthkit imports if not installed."""
    import importlib
    import sys
    import types

    def stub(name, **attrs):
        try:
            importlib.import_module(name)
        except ImportError:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            monkeypatch.setitem(sys.modules, name, mod)

    stub("anemoi")
    stub("anemoi.datasets")
    stub("anemoi.datasets.data")
    stub("anemoi.datasets.data.dataset", Dataset=object)
    stub("earthkit")
    stub("earthkit.data")
    stub("earthkit.data.utils")
    stub("earthkit.data.utils.dates", to_datetime=lambda x: x)
    sys.modules.pop("weathergen.readers_extra.data_reader_anemoi_rt", None)
    return importlib.import_module("weathergen.readers_extra.data_reader_anemoi_rt")


class _FakeDataset:
    """ds[0, idxs] -> (1, len(idxs), N) like anemoi's (time, variable, ensemble?, point) read"""

    def __init__(self, n_points, seed=0):
        self.data = np.random.default_rng(seed).normal(size=(40, n_points)).astype(np.float32)
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        _, idxs = key
        return self.data[idxs][None]


def _reference_get(reader, t_idxs, dtr, fake_dynamic):
    """Verbatim copy of the original DataReaderAnemoiRT._get body (before caching)."""
    latlon = np.concatenate(
        [np.expand_dims(reader.latitudes, 0), np.expand_dims(reader.longitudes, 0)], axis=0
    ).transpose()
    coords = np.vstack(list((latlon,) * len(t_idxs)))
    datetimes = []
    t_cur = dtr.start
    while t_cur < dtr.end:
        datetimes += [t_cur]
        t_cur += reader.frequency
    geoinfos_static = reader.ds[0, list(reader.geoinfo_idx_static)][0].transpose()
    geoinfos_static = np.concatenate([geoinfos_static for _ in t_idxs])
    geoinfos_dynamic = fake_dynamic(
        datetimes, reader.latitudes, reader.longitudes, reader.geoinfo_channels_dynamic
    )
    geoinfos = np.empty((coords.shape[0], len(reader.geoinfo_idx)))
    for idx, i in enumerate(reader.geoinfo_idx_static_lin):
        geoinfos[:, i] = geoinfos_static[:, idx]
    for i, ch in zip(reader.geoinfo_idx_dynamic_lin, reader.geoinfo_channels_dynamic, strict=True):
        geoinfos[:, i] = geoinfos_dynamic[ch]
    data = np.empty((0, 3))
    temp = np.repeat(np.array([datetimes], dtype=np.datetime64), len(reader.latitudes), axis=0)
    return coords, geoinfos, data, temp.transpose().flatten()


def test_reader_grid_cache_equals_original(monkeypatch):
    import types

    mod = _import_reader_module(monkeypatch)
    n_points = 500
    reader = object.__new__(mod.DataReaderAnemoiRT)
    rng = np.random.default_rng(1)
    reader.latitudes = rng.uniform(-90, 90, n_points).astype(np.float32)
    reader.longitudes = rng.uniform(-180, 180, n_points).astype(np.float32)
    reader.frequency = np.timedelta64(1, "h")
    reader.ds = _FakeDataset(n_points)
    # layout like ERA5: 13 geoinfo channels of which 5 dynamic (interleaved with static ones)
    dynamic_lin = [4, 5, 6, 7, 8]
    static_lin = [0, 1, 2, 3, 9, 10, 11, 12]
    reader.geoinfo_idx = list(range(13))
    reader.geoinfo_idx_static_lin = static_lin
    reader.geoinfo_idx_static = [20 + i for i in static_lin]
    reader.geoinfo_idx_dynamic_lin = dynamic_lin
    reader.geoinfo_channels_dynamic = ["insolation", "cos_local_time", "sin_local_time",
                                       "cos_julian_day", "sin_julian_day"]  # fmt: skip
    reader._grid_cache = {}
    reader._grid_confirmed = set()
    reader._grid_disabled = set()
    reader._grid_cache_option = None
    reader._grid_cache_verify_option = None

    def fake_dynamic(times, lats, lons, selection):
        out = {k: [] for k in selection}
        for t in times:
            h = float((t - np.datetime64("2020-01-01T00:00:00")) / np.timedelta64(1, "h"))
            for j, k in enumerate(selection):
                out[k].append(np.sin(lats * (j + 1) + lons * 0.01 + h).astype(np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}

    monkeypatch.setattr(mod, "_anemoi_get_dynamic_forcings", fake_dynamic)

    for enabled in ["1", "0"]:
        monkeypatch.setenv("WEATHERGEN_GRID_CACHE", enabled)
        reader._grid_cache = {}
        reader._grid_confirmed = set()
        reader._grid_disabled = set()
        for step in range(5):
            start = T0 + step * STEP
            dtr = types.SimpleNamespace(start=start, end=start + STEP)
            t_idxs = np.arange(6)
            monkeypatch.setattr(reader, "_get_dataset_idxs", lambda idx, t=t_idxs, d=dtr: (t, d),
                                raising=False)  # fmt: skip
            got = reader._get(0, [0, 1, 2])
            want = _reference_get(reader, t_idxs, dtr, fake_dynamic)
            for name, g, w in zip(["coords", "geoinfos", "data", "datetimes"],
                                  [got.coords, got.geoinfos, got.data, got.datetimes],
                                  want, strict=True):  # fmt: skip
                assert g.dtype == w.dtype and g.shape == w.shape, name
                assert np.array_equal(g, w), name
            # mutate the result in place: must not leak into the cache
            got.coords *= 0
            got.geoinfos *= 0
    # static data was read only once with the cache on (plus once per call for the reference)
    reader._grid_cache = {}


def test_verify_mode_passes_and_detects_corruption():
    latlon = _grid()
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    tok = TokenizerMasking(HL, None, grid_cache_option=True, grid_cache_verify_option=True)
    for step in range(5):
        rdata, time_win = _window(step, latlon)
        _run(tok, rdata, time_win, cell_mask)

    # corrupt the cached template: the next hit must be flagged
    entry = tok._target_coords_cache[STREAM_INFO["stream_id"]]
    entry.coords_local[3, 40] += 1.0
    rdata, time_win = _window(5, latlon)
    with pytest.raises(AssertionError, match="grid cache mismatch"):
        _run(tok, rdata, time_win, cell_mask)


# --------------------------------------------------------------------------------------------
# constructor / option switches (standalone, no HEALPix work beyond Tokenizer init)
# --------------------------------------------------------------------------------------------


def test_grid_cache_switches(monkeypatch):
    monkeypatch.delenv("WEATHERGEN_GRID_CACHE", raising=False)
    monkeypatch.delenv("WEATHERGEN_GRID_CACHE_VERIFY", raising=False)
    assert grid_cache_enabled() is True
    assert grid_cache_verify() is False
    monkeypatch.setenv("WEATHERGEN_GRID_CACHE", "0")
    monkeypatch.setenv("WEATHERGEN_GRID_CACHE_VERIFY", "1")
    assert grid_cache_enabled() is False
    assert grid_cache_verify() is True
    # explicit option wins over the environment
    assert grid_cache_enabled(True) is True
    assert grid_cache_enabled(False) is False
    assert grid_cache_verify(False) is False
    assert grid_cache_verify(True) is True


def test_constructor_options_and_verify_name():
    """Must not raise TypeError (the old `grid_cache_verify` parameter shadowed the function)."""
    on = TokenizerMasking(HL, None, grid_cache_option=True, grid_cache_verify_option=True)
    off = TokenizerMasking(HL, None, grid_cache_option=False, grid_cache_verify_option=False)
    assert on._grid_cache_enabled and on._grid_cache_verify
    assert not off._grid_cache_enabled and not off._grid_cache_verify


def test_assert_identical_accepts_equal_and_rejects_unequal():
    _assert_identical("t", torch.ones(3), torch.ones(3))
    _assert_identical("n", np.arange(4), np.arange(4))
    _assert_identical("nested", [torch.zeros(2), np.ones(2)], [torch.zeros(2), np.ones(2)])
    _assert_identical("none", None, None)
    with pytest.raises(AssertionError, match="grid cache mismatch"):
        _assert_identical("t", torch.ones(3), torch.zeros(3))
    with pytest.raises(AssertionError, match="grid cache mismatch"):
        _assert_identical("none", None, torch.zeros(1))


def test_template_clone_does_not_alias():
    template = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    times = torch.tensor([[100.0, 101.0], [200.0, 201.0]])
    geo = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = target_coords_from_template(template, geo, times)
    assert torch.equal(got[:, 1:3], times) and torch.equal(got[:, 3:6], geo)
    assert torch.equal(got[:, 0], template[:, 0]) and torch.equal(got[:, 6:], template[:, 6:])
    got[0, 0] = -99
    assert template[0, 0] == 0  # clone, not a view


def test_pad_tokens_and_stream_id_are_separate_cache_keys():
    tok = TokenizerMasking(HL, None, grid_cache_option=True)
    rdata, _ = _window(0, _grid())
    tok.get_tokens_windows(STREAM_INFO, [rdata], False)
    tok.get_tokens_windows(STREAM_INFO, [rdata], True)
    other = dict(STREAM_INFO, stream_id=99)
    tok.get_tokens_windows(other, [rdata], False)
    keys = set(tok._tokens_cache)
    assert (1, 32, False, HL) in keys
    assert (1, 32, True, HL) in keys
    assert (99, 32, False, HL) in keys
    # a stream with no stream_id is never cached
    no_id = dict(STREAM_INFO)
    no_id.pop("stream_id")
    before = len(tok._tokens_cache)
    tok.get_tokens_windows(no_id, [rdata], False)
    assert len(tok._tokens_cache) == before


def test_target_coords_cache_fills_on_second_matching_window():
    cached, _ = _tokenizers()
    latlon = _grid()
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    r0, tw0 = _window(0, latlon)
    r1, tw1 = _window(1, latlon)
    r2, tw2 = _window(2, latlon)
    _run(cached, r0, tw0, cell_mask)
    entry = cached._target_coords_cache[STREAM_INFO["stream_id"]]
    assert entry.coords_local is None, "first window only registers the key"
    _run(cached, r1, tw1, cell_mask)
    assert entry.coords_local is not None, "second matching window stores outputs"
    t2 = cached.get_tokens_windows(STREAM_INFO, [r2], False)[0]
    cached.get_target_coords(STREAM_INFO, r2, t2, tw2, cell_mask)
    # third window is a hit: idxs_cells object identity is reused
    assert cached.get_tokens_windows(STREAM_INFO, [_window(3, latlon)[0]], False)[0][0] is t2[0]


def test_geoinfo_width_change_falls_back_to_regular():
    """A window with a different number of geoinfo columns must not reuse the template."""
    cached, plain = _tokenizers()
    latlon = _grid()
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    for step in range(3):
        rdata_c, time_win = _window(step, latlon)
        rdata_p, _ = _window(step, latlon)
        tok_c, res_c = _run(cached, rdata_c, time_win, cell_mask)
        tok_p, res_p = _run(plain, rdata_p, time_win, cell_mask)
        _tokens_same(tok_c, tok_p)
        _assert_same(res_c, res_p)
    rdata_c, time_win = _window(3, latlon)
    rdata_p, _ = _window(3, latlon)
    rdata_c.geoinfos = np.concatenate([rdata_c.geoinfos, rdata_c.geoinfos[:, :1]], axis=1)
    rdata_p.geoinfos = np.concatenate([rdata_p.geoinfos, rdata_p.geoinfos[:, :1]], axis=1)
    _, res_c = _run(cached, rdata_c, time_win, cell_mask)
    _, res_p = _run(plain, rdata_p, time_win, cell_mask)
    _assert_same(res_c, res_p)


def test_tokenization_window1_mismatch_drops_cache(caplog):
    tok = TokenizerMasking(HL, None, grid_cache_option=True)
    latlon = _grid()
    cache_key = (STREAM_INFO["stream_id"], 32, False, HL)
    tok.get_tokens_windows(STREAM_INFO, [_window(0, latlon)[0]], False)
    entry = tok._tokens_cache[cache_key]
    scrambled = False
    for cell in entry.idxs_cells:
        for idxs in cell:
            if idxs.numel() > 0:
                idxs[0] += 1
                scrambled = True
                break
        if scrambled:
            break
    assert scrambled, "test grid produced no token indices to corrupt"
    with caplog.at_level(logging.WARNING):
        tok.get_tokens_windows(STREAM_INFO, [_window(1, latlon)[0]], False)
    assert "Grid cache dropped for tokenization" in caplog.text
    assert cache_key in tok._tokens_disabled
    assert cache_key not in tok._tokens_cache
    # later windows still run (uncached) and match a plain tokenizer
    plain = TokenizerMasking(HL, None, grid_cache_option=False)
    t_c = tok.get_tokens_windows(STREAM_INFO, [_window(2, latlon)[0]], False)[0]
    t_p = plain.get_tokens_windows(STREAM_INFO, [_window(2, latlon)[0]], False)[0]
    _tokens_same(t_c, t_p)


def test_tokenization_verify_after_confirm_detects_corruption():
    tok = TokenizerMasking(HL, None, grid_cache_option=True, grid_cache_verify_option=True)
    latlon = _grid()
    tok.get_tokens_windows(STREAM_INFO, [_window(0, latlon)[0]], False)
    tok.get_tokens_windows(STREAM_INFO, [_window(1, latlon)[0]], False)  # confirm
    entry = tok._tokens_cache[(STREAM_INFO["stream_id"], 32, False, HL)]
    scrambled = False
    for cell in entry.idxs_cells:
        for idxs in cell:
            if idxs.numel() > 0:
                idxs[0] += 1
                scrambled = True
                break
        if scrambled:
            break
    assert scrambled
    with pytest.raises(AssertionError, match="grid cache mismatch"):
        tok.get_tokens_windows(STREAM_INFO, [_window(2, latlon)[0]], False)


def test_target_coords_window1_mismatch_drops_cache(caplog):
    cached, plain = _tokenizers()
    latlon = _grid()
    cell_mask = np.ones(12 * 4**HL, dtype=bool)
    r0, tw0 = _window(0, latlon)
    _run(cached, r0, tw0, cell_mask)
    entry = cached._target_coords_cache[STREAM_INFO["stream_id"]]
    assert entry.geometry_digest is not None and entry.coords_local is None
    entry.geometry_digest = b"\x00" * 32
    r1, tw1 = _window(1, latlon)
    with caplog.at_level(logging.WARNING):
        _, res_c = _run(cached, r1, tw1, cell_mask)
    assert "Grid cache dropped for target coords" in caplog.text
    assert STREAM_INFO["stream_id"] in cached._target_coords_disabled
    assert STREAM_INFO["stream_id"] not in cached._target_coords_cache
    r1p, _ = _window(1, latlon)
    _, res_p = _run(plain, r1p, tw1, cell_mask)
    _assert_same(res_c, res_p)


def _make_reader(mod, n_points=500, seed=1, grid_cache=None, grid_cache_verify=None):
    reader = object.__new__(mod.DataReaderAnemoiRT)
    rng = np.random.default_rng(seed)
    reader.latitudes = rng.uniform(-90, 90, n_points).astype(np.float32)
    reader.longitudes = rng.uniform(-180, 180, n_points).astype(np.float32)
    reader.frequency = np.timedelta64(1, "h")
    reader.ds = _FakeDataset(n_points)
    dynamic_lin = [4, 5, 6, 7, 8]
    static_lin = [0, 1, 2, 3, 9, 10, 11, 12]
    reader.geoinfo_idx = list(range(13))
    reader.geoinfo_idx_static_lin = static_lin
    reader.geoinfo_idx_static = [20 + i for i in static_lin]
    reader.geoinfo_idx_dynamic_lin = dynamic_lin
    reader.geoinfo_channels_dynamic = [
        "insolation",
        "cos_local_time",
        "sin_local_time",
        "cos_julian_day",
        "sin_julian_day",
    ]
    reader._grid_cache = {}
    reader._grid_confirmed = set()
    reader._grid_disabled = set()
    reader._grid_cache_option = grid_cache
    reader._grid_cache_verify_option = grid_cache_verify
    return reader


def _fake_dynamic(times, lats, lons, selection):
    out = {k: [] for k in selection}
    for t in times:
        h = float((t - np.datetime64("2020-01-01T00:00:00")) / np.timedelta64(1, "h"))
        for j, k in enumerate(selection):
            out[k].append(np.sin(lats * (j + 1) + lons * 0.01 + h).astype(np.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


def test_reader_static_data_is_read_once_when_cached(monkeypatch):
    import types

    mod = _import_reader_module(monkeypatch)
    monkeypatch.setattr(mod, "_anemoi_get_dynamic_forcings", _fake_dynamic)
    reader = _make_reader(mod, grid_cache=True)

    def call(step, hours=6):
        start = T0 + step * STEP
        dtr = types.SimpleNamespace(start=start, end=start + hours * np.timedelta64(1, "h"))
        t_idxs = np.arange(hours)
        monkeypatch.setattr(
            reader, "_get_dataset_idxs", lambda idx, t=t_idxs, d=dtr: (t, d), raising=False
        )
        return reader._get(0, [0, 1, 2])

    call(0)
    assert reader.ds.reads == 1
    call(1)  # window 1 rebuilds once to compare with window 0
    assert reader.ds.reads == 2
    call(2)
    assert reader.ds.reads == 2
    assert 6 in reader._grid_confirmed


def test_reader_stream_option_false_disables_cache(monkeypatch):
    import types

    mod = _import_reader_module(monkeypatch)
    monkeypatch.setattr(mod, "_anemoi_get_dynamic_forcings", _fake_dynamic)
    reader = _make_reader(mod, grid_cache=False)

    def call(step):
        start = T0 + step * STEP
        dtr = types.SimpleNamespace(start=start, end=start + STEP)
        t_idxs = np.arange(6)
        monkeypatch.setattr(
            reader, "_get_dataset_idxs", lambda idx, t=t_idxs, d=dtr: (t, d), raising=False
        )
        return reader._get(0, [0, 1, 2])

    call(0)
    call(1)
    call(2)
    assert reader.ds.reads == 3
    assert reader._grid_cache == {}


def test_reader_window_length_is_the_cache_key(monkeypatch):
    import types

    mod = _import_reader_module(monkeypatch)
    monkeypatch.setattr(mod, "_anemoi_get_dynamic_forcings", _fake_dynamic)
    reader = _make_reader(mod, grid_cache=True)

    def call(hours):
        dtr = types.SimpleNamespace(start=T0, end=T0 + hours * np.timedelta64(1, "h"))
        t_idxs = np.arange(hours)
        monkeypatch.setattr(
            reader, "_get_dataset_idxs", lambda idx, t=t_idxs, d=dtr: (t, d), raising=False
        )
        return reader._get(0, [0, 1, 2])

    a = call(6)
    b = call(3)
    c = call(6)
    assert a.coords.shape[0] == 6 * 500 and b.coords.shape[0] == 3 * 500
    assert set(reader._grid_cache) == {3, 6}
    # first length-6 build + length-3 build + length-6 confirmation rebuild
    assert reader.ds.reads == 3
    assert np.array_equal(a.coords, c.coords)


def test_reader_verify_mode_and_empty_window(monkeypatch):
    import types

    mod = _import_reader_module(monkeypatch)
    monkeypatch.setattr(mod, "_anemoi_get_dynamic_forcings", _fake_dynamic)
    reader = _make_reader(mod, grid_cache=True, grid_cache_verify=True)

    dtr = types.SimpleNamespace(start=T0, end=T0 + STEP)
    t_idxs = np.arange(6)
    monkeypatch.setattr(
        reader, "_get_dataset_idxs", lambda idx, t=t_idxs, d=dtr: (t, d), raising=False
    )
    reader._get(0, [0, 1, 2])
    reader._get(0, [0, 1, 2])  # hit + verify against a fresh _build_grid

    reader._grid_cache[6].coords[0, 0] += 1.0
    with pytest.raises(AssertionError, match="grid cache mismatch"):
        reader._get(0, [0, 1, 2])

    empty = _make_reader(mod, grid_cache=True)
    monkeypatch.setattr(
        empty,
        "_get_dataset_idxs",
        lambda idx: (np.array([], dtype=np.int64), dtr),
        raising=False,
    )
    rd = empty._get(0, [0, 1, 2])
    assert rd.coords.shape[0] == 0 and empty._grid_cache == {}


def test_reader_window1_mismatch_drops_cache(monkeypatch, caplog):
    import types

    mod = _import_reader_module(monkeypatch)
    monkeypatch.setattr(mod, "_anemoi_get_dynamic_forcings", _fake_dynamic)
    reader = _make_reader(mod, grid_cache=True)

    def call(step):
        start = T0 + step * STEP
        dtr = types.SimpleNamespace(start=start, end=start + STEP)
        t_idxs = np.arange(6)
        monkeypatch.setattr(
            reader, "_get_dataset_idxs", lambda idx, t=t_idxs, d=dtr: (t, d), raising=False
        )
        return reader._get(0, [0, 1, 2])

    call(0)
    reader._grid_cache[6].coords[0, 0] += 1.0
    with caplog.at_level(logging.WARNING):
        got = call(1)
    assert "Grid cache dropped for anemoi_rt" in caplog.text
    assert 6 in reader._grid_disabled
    assert reader._grid_cache == {}
    assert got.coords.shape[0] == 6 * 500
    call(2)
    assert reader.ds.reads == 3  # uncached from window 1 on


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
