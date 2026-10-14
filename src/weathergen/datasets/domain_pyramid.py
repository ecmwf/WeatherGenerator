# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Multi-resolution HEALPix latent support: a *pyramid* of `Domain` objects.

The single-level `Domain` (domain.py) maps global nested HEALPix indices to a compact
[0, num_active) indexing for ONE healpix level, optionally cropped to a bounding box.

A multi-resolution latent needs several such domains at different levels (e.g. a
global hl5 domain for reanalysis streams and a Nordic hl8 domain for a high-resolution
regional stream), plus the exact geometric maps that move information between levels:

    coarse compact cell  --(children)-->  fine compact cells    (prolongation)
    fine   compact cell  --(parent)--->   coarse compact cell   (restriction)

In the NESTED scheme refining by one level splits a cell into 4 children whose indices
are the parent index shifted left by 2 bits plus {0, 1, 2, 3}. From level Lc to a finer
level Lf:

    parent(g_f)   = g_f // 4**(Lf - Lc)
    children(g_c) = g_c * 4**(Lf - Lc) + [0 .. 4**(Lf - Lc) - 1]

so the cross-level operators are exact index gathers, not interpolations. Every map is
restricted to the overlap of the two domains: a fine cell whose parent is not active at
the coarse level, or a coarse child that is not active at the fine level, is recorded
as -1 ("unpaired") instead of being mismapped.

This module owns geometry only. Which stream attaches to which level is decided in
`stream_levels.py`; the latent exchange operators live in `model/latent_cascade.py`.

Config
------
Without a `latent_levels` block the pyramid has one level, built by
`Domain.from_config(cf)` (top-level `healpix_level` + optional `domain:` block), i.e.
exactly the single-latent behaviour.

.. code-block:: yaml

    latent_levels:
      - level: 5                  # coarse level, global (no domain block)
      - level: 8                  # fine level, cropped
        domain: {lon_min: -10.0, lon_max: 35.0, lat_min: 50.0, lat_max: 75.0, pad_rings: 1}
"""

import logging
import warnings

import astropy_healpix as hp
import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.domain import Domain

_logger = logging.getLogger(__name__)

# the pyramid is built from the config in several places (sampler, model, encoder, model
# params); cache it so the construction and its log lines happen once per process
_PYRAMID_CACHE: dict[tuple, "DomainPyramid"] = {}


def cells_parent(global_idxs: NDArray, level_from: int, level_to: int) -> NDArray[np.int64]:
    """Global nested index of the ancestor at the coarser level `level_to` (<= level_from)."""
    assert level_to <= level_from, "parent must be at a coarser (<=) level"
    factor = 4 ** (level_from - level_to)
    return np.asarray(global_idxs, dtype=np.int64) // factor


def cells_children(global_idxs: NDArray, level_from: int, level_to: int) -> NDArray[np.int64]:
    """
    Global nested indices of the descendants at the finer level `level_to` (>= level_from).

    Returns shape (len(global_idxs), 4**(level_to - level_from)); row i lists the children
    of global_idxs[i] in ascending order.
    """
    assert level_to >= level_from, "children must be at a finer (>=) level"
    n_children = 4 ** (level_to - level_from)
    base = np.asarray(global_idxs, dtype=np.int64)[:, None] * n_children
    return base + np.arange(n_children, dtype=np.int64)[None, :]


class DomainPyramid:
    """
    An ordered set of `Domain` objects at increasing healpix level, with exact
    parent/child compact-index maps between adjacent configured levels.

    Attributes
    ----------
    levels : list[int]
        The configured healpix levels, sorted ascending (coarse -> fine).
    domains : dict[int, Domain]
        `domains[L]` is the `Domain` at level L.

    Torch-free and picklable (numpy arrays and scalars only), like `Domain`.
    """

    def __init__(self, domains: dict[int, Domain]) -> None:
        assert len(domains) >= 1, "DomainPyramid needs at least one level"
        self.levels: list[int] = sorted(domains.keys())
        self.domains: dict[int, Domain] = dict(domains)
        for lvl, dom in self.domains.items():
            assert dom.healpix_level == lvl, f"domain for level {lvl} has level {dom.healpix_level}"

        # maps between every ADJACENT configured level pair, keyed by (coarse, fine)
        self._child_of_parent: dict[tuple[int, int], NDArray] = {}
        self._parent_of_child: dict[tuple[int, int], NDArray] = {}
        self._child_valid: dict[tuple[int, int], NDArray] = {}
        for lc, lf in zip(self.levels[:-1], self.levels[1:], strict=True):
            self._build_level_pair(lc, lf)

        # ancestor maps between arbitrary (not only adjacent) level pairs, built lazily
        self._ancestor: dict[tuple[int, int], NDArray] = {}
        # k-ring neighbourhoods per level, built lazily
        self._rings: dict[tuple[int, int], NDArray] = {}

    # -----------------------------------------------------------------------
    # constructors
    # -----------------------------------------------------------------------

    @classmethod
    def single(cls, domain: Domain) -> "DomainPyramid":
        """One-level pyramid wrapping `domain` (single-latent behaviour, no cross-level maps)."""
        return cls({domain.healpix_level: domain})

    @classmethod
    def from_config(cls, cf) -> "DomainPyramid":
        """
        Build the pyramid from the optional `latent_levels` config block (see module doc).
        Without it, a one-level pyramid around `Domain.from_config(cf)`.
        """
        levels_cfg = cf.get("latent_levels", None)
        if not levels_cfg:
            return cls.single(Domain.from_config(cf))

        domains: dict[int, Domain] = {}
        for entry in levels_cfg:
            lvl = int(entry["level"])
            assert lvl not in domains, f"latent_levels: level {lvl} is listed twice"
            domains[lvl] = Domain.from_level_config(lvl, entry.get("domain", None))

        if cf.get("domain", None) is not None:
            _logger.warning(
                "Both `latent_levels` and a top-level `domain` block are configured; the "
                "top-level `domain` is ignored, each level uses its own `domain` entry."
            )
        if cf.get("healpix_level", None) not in domains:
            _logger.warning(
                "healpix_level=%s is not one of the latent levels %s; it is no longer used "
                "for the latent grid (the latent levels are).",
                cf.get("healpix_level", None),
                sorted(domains),
            )
        return cls(domains)

    # -----------------------------------------------------------------------
    # level access
    # -----------------------------------------------------------------------

    @property
    def coarsest(self) -> int:
        return self.levels[0]

    @property
    def finest(self) -> int:
        return self.levels[-1]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def is_single(self) -> bool:
        return len(self.levels) == 1

    def domain(self, level: int) -> Domain:
        return self.domains[level]

    def __len__(self) -> int:
        return len(self.levels)

    def __repr__(self) -> str:
        parts = ", ".join(f"hl{lvl}:{len(self.domains[lvl])}" for lvl in self.levels)
        return f"DomainPyramid({parts})"

    # -----------------------------------------------------------------------
    # cross-level maps (compact indexing)
    # -----------------------------------------------------------------------

    def _build_level_pair(self, lc: int, lf: int) -> None:
        """
        Parent<->child compact maps for the adjacent pair (coarse lc, fine lf):

        parent_of_child[(lc, lf)] : (num_fine,)  compact parent of each fine cell, or -1
        child_of_parent[(lc, lf)] : (num_coarse, 4**(lf-lc))  compact children, -1 if inactive
        child_valid[(lc, lf)]     : (num_coarse, 4**(lf-lc))  bool, child_of_parent >= 0
        """
        dom_c = self.domains[lc]
        dom_f = self.domains[lf]
        n_children = 4 ** (lf - lc)

        parent_compact = dom_c.to_compact_safe(cells_parent(dom_f.active_cells, lf, lc))
        self._parent_of_child[(lc, lf)] = parent_compact

        children_global = cells_children(dom_c.active_cells, lc, lf)
        children_compact = dom_f.to_compact_safe(children_global.reshape(-1)).reshape(
            children_global.shape
        )
        self._child_of_parent[(lc, lf)] = children_compact
        self._child_valid[(lc, lf)] = children_compact >= 0

        _logger.info(
            "DomainPyramid hl%d<->hl%d: coarse=%d fine=%d children/parent=%d | fine cells "
            "with an active parent: %d/%d | coarse cells with >=1 active child: %d/%d",
            lc,
            lf,
            len(dom_c),
            len(dom_f),
            n_children,
            int((parent_compact >= 0).sum()),
            len(dom_f),
            int((children_compact >= 0).any(axis=1).sum()),
            len(dom_c),
        )

    def _pair_key(self, lc: int, lf: int) -> tuple[int, int]:
        assert (lc, lf) in self._child_of_parent, (
            f"({lc},{lf}) is not an adjacent configured level pair; "
            f"configured pairs: {list(self._child_of_parent.keys())}"
        )
        return (lc, lf)

    def parent_of_child(self, coarse_level: int, fine_level: int) -> NDArray[np.int64]:
        """(num_fine,) compact parent index per fine cell, -1 if unpaired."""
        return self._parent_of_child[self._pair_key(coarse_level, fine_level)]

    def child_of_parent(self, coarse_level: int, fine_level: int) -> NDArray[np.int64]:
        """(num_coarse, 4**(lf-lc)) compact child indices, -1 where inactive."""
        return self._child_of_parent[self._pair_key(coarse_level, fine_level)]

    def child_valid(self, coarse_level: int, fine_level: int) -> NDArray[np.bool_]:
        """(num_coarse, 4**(lf-lc)) bool mask of valid children."""
        return self._child_valid[self._pair_key(coarse_level, fine_level)]

    def ancestor_of(self, fine_level: int, coarse_level: int) -> NDArray[np.int64]:
        """
        (num_cells(fine_level),) compact index of each fine cell's ancestor at
        `coarse_level` (any two configured levels, coarse_level <= fine_level), -1 if the
        ancestor is outside the coarse domain. Used by the multi-level decoder.
        """
        assert coarse_level <= fine_level
        key = (coarse_level, fine_level)
        if key not in self._ancestor:
            dom_f, dom_c = self.domains[fine_level], self.domains[coarse_level]
            self._ancestor[key] = dom_c.to_compact_safe(
                cells_parent(dom_f.active_cells, fine_level, coarse_level)
            )
        return self._ancestor[key]

    def rings(self, level: int, num_rings: int) -> NDArray[np.int64]:
        """
        (num_cells(level), K) compact indices of the cells within `num_rings` HEALPix
        neighbour steps of each active cell at `level` (the cell itself included), padded
        with -1. Ring 0 is the cell alone, 1 ring adds its 8 neighbours (up to 9 cells),
        2 rings up to 25, ... Rings are grown on the full sphere and then restricted to the
        level's domain, so cells outside the domain are left out (fewer valid entries near
        a domain edge or a HEALPix corner).
        """
        assert num_rings >= 0
        key = (level, num_rings)
        if key not in self._rings:
            dom = self.domains[level]
            nside = 2**level
            big = np.iinfo(np.int64).max
            sets = dom.active_cells[:, None].astype(np.int64)  # (n, 1) global indices
            for _ in range(num_rings):
                # pad entries are replaced by the row's own cell before the lookup
                safe = np.where(sets == big, sets[:, :1], sets)
                with warnings.catch_warnings(action="ignore"):
                    nb = hp.neighbours(safe.ravel(), nside, order="nested")  # (8, n * K)
                nb = nb.T.reshape(len(sets), -1)
                nb = np.where(nb < 0, sets[:, :1], nb)  # missing healpix neighbours
                sets = _unique_rows(np.concatenate([safe, nb], axis=1), big)
            compact = dom.to_compact_safe(np.where(sets == big, -1, sets))
            # valid entries first, then the -1 padding
            order = np.argsort(compact < 0, axis=1, kind="stable")
            compact = np.take_along_axis(compact, order, axis=1)
            width = max(int((compact >= 0).sum(axis=1).max()), 1)
            self._rings[key] = compact[:, :width]
        return self._rings[key]


def _unique_rows(a: NDArray[np.int64], pad: int) -> NDArray[np.int64]:
    """Per-row unique values of a non-negative int array, padded on the right with `pad`."""
    a = np.sort(a, axis=1)
    dup = np.zeros(a.shape, dtype=bool)
    dup[:, 1:] = a[:, 1:] == a[:, :-1]
    a = np.sort(np.where(dup, pad, a), axis=1)
    width = int((a != pad).sum(axis=1).max())
    return a[:, :width]


def _pyramid_cache_key(cf) -> tuple:
    def _dom_key(d):
        if d is None or not d.get("enabled", True):
            return None
        keys = ("lon_min", "lon_max", "lat_min", "lat_max")
        return tuple(float(d[k]) for k in keys) + (int(d.get("pad_rings", 0)),)

    levels_cfg = cf.get("latent_levels", None)
    if not levels_cfg:
        return ("single", int(cf.healpix_level), _dom_key(cf.get("domain", None)))
    return tuple(sorted((int(e["level"]), _dom_key(e.get("domain", None))) for e in levels_cfg))


def build_domain_pyramid(cf) -> DomainPyramid:
    """
    Single entry point for the latent-domain pyramid (cached per configuration).

    With no `latent_levels` block this is a one-level pyramid whose only domain is
    `Domain.from_config(cf)`, so all single-level behaviour is unchanged.
    """
    key = _pyramid_cache_key(cf)
    pyramid = _PYRAMID_CACHE.get(key)
    if pyramid is None:
        pyramid = DomainPyramid.from_config(cf)
        _PYRAMID_CACHE[key] = pyramid
    return pyramid


def model_domain(cf) -> Domain:
    """
    Domain of the finest latent level: the latent that drives the latent outputs
    (latent_state, SSL heads, latent Zarr writer). Equals `Domain.from_config(cf)` for a
    single-level run.
    """
    pyramid = build_domain_pyramid(cf)
    return pyramid.domain(pyramid.finest)
