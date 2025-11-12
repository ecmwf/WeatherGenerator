# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.common.io import IOReaderData


class StreamData:
    """
    StreamData object that encapsulates all data the model ingests for one batch item
    for one stream.
    """

    def __init__(self, idx: int, forecast_steps: int, healpix_cells: int) -> None:
        """
        StreamData object

        Parameters
        ----------
        forecast_steps : int
            Number of forecast steps
        healpix_cells : int
            Number of healpix cells for source

        Returns
        -------
        None
        """

        self.mask_value = 0.0

        self.forecast_steps = forecast_steps
        self.healpix_cells = healpix_cells

        self.source_is_spoof = False
        self.target_is_spoof = False

        # initialize empty members
        self.sample_idx = idx
        self.target_coords = [torch.tensor([]) for _ in range(forecast_steps + 1)]
        self.target_coords_raw = [[] for _ in range(forecast_steps + 1)]
        self.target_times_raw = [[] for _ in range(forecast_steps + 1)]
        # this is not directly used but to precompute index in compute_idxs_predict()
        self.target_coords_lens = [
            torch.tensor([0 for _ in range(self.healpix_cells)]) for _ in range(forecast_steps + 1)
        ]
        self.target_tokens = [torch.tensor([]) for _ in range(forecast_steps + 1)]
        self.target_tokens_lens = [
            torch.tensor([0 for _ in range(self.healpix_cells)]) for _ in range(forecast_steps + 1)
        ]

        # source tokens per cell
        self.source_tokens_cells = []
        # length of source tokens per cell (without padding)
        self.source_tokens_lens = []
        self.source_centroids = []
        # unprocessed source (for logging)
        self.source_raw = []
        # auxiliary data for scatter operation that changes from stream-centric to cell-centric
        # processing after embedding
        self.source_idxs_embed = torch.tensor([])
        self.source_idxs_embed_pe = torch.tensor([])

    def to_device(self, device: str) -> None:
        """
        Move data to GPU

        Parameters
        ----------
        device : str
            Device the data is moved/mapped to.

        Returns
        -------
        None
        """

        self.source_tokens_cells = self.source_tokens_cells.to(device, non_blocking=True)
        self.source_centroids = self.source_centroids.to(device, non_blocking=True)
        self.source_tokens_lens = self.source_tokens_lens.to(device, non_blocking=True)

        self.target_coords = [t.to(device, non_blocking=True) for t in self.target_coords]
        self.target_tokens = [t.to(device, non_blocking=True) for t in self.target_tokens]
        self.target_tokens_lens = [t.to(device, non_blocking=True) for t in self.target_tokens_lens]

        self.source_idxs_embed = self.source_idxs_embed.to(device, non_blocking=True)
        self.source_idxs_embed_pe = self.source_idxs_embed_pe.to(device, non_blocking=True)

        self.target_srclk_raw = [t.to(device, non_blocking=True) for t in self.target_srclk_raw]
        self.target_srclk_tokens_lens = [
            t.to(device, non_blocking=True) for t in self.target_srclk_tokens_lens
        ]
        self.target_srclk_tokens_cells = [
            t.to(device, non_blocking=True) for t in self.target_srclk_tokens_cells
        ]
        self.target_srclk_centroids = [
            t.to(device, non_blocking=True) for t in self.target_srclk_centroids
        ]
        
        self.target_srclk_idxs_embed = [
            t.to(device, non_blocking=True) for t in self.target_srclk_idxs_embed
        ]
        self.target_srclk_idxs_embed_pe = [
            t.to(device, non_blocking=True) for t in self.target_srclk_idxs_embed_pe
        ]

        return self

    def add_empty_source(self, source: IOReaderData) -> None:
        """
        Add an empty source for an input.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        source = spoof(source)
        self.source_raw += [source]
        self.source_tokens_lens += [torch.ones([self.healpix_cells], dtype=torch.int32)]
        self.source_tokens_cells += [torch.tensor([])]
        self.source_centroids += [torch.tensor([])]

    def add_empty_target_srclk(self, fstep: int) -> None:
        """
        Add an empty target for an input encoded like source.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """

        self.target_srclk_raw[fstep] += [torch.tensor([])]
        self.target_srclk_tokens_lens[fstep] += [torch.zeros([self.nhc_source], dtype=torch.int32)]
        self.target_srclk_tokens_cells[fstep] += [torch.tensor([])]
        self.target_srclk_centroids[fstep] += [torch.tensor([])]

    def add_empty_target(self, fstep: int) -> None:
        """
        Add an empty target for an input.

        Parameters
        ----------
        fstep : int
            forecast step

        Returns
        -------
        None
        """

        self.target_tokens[fstep] += [torch.tensor([], dtype=torch.int32)]
        self.target_tokens_lens[fstep] += [torch.zeros([self.healpix_cells], dtype=torch.int32)]
        self.target_coords[fstep] += [torch.zeros((0, 105)) for _ in range(self.healpix_cells)]
        self.target_coords_lens[fstep] += [torch.zeros([self.healpix_cells], dtype=torch.int32)]
        self.target_coords_raw[fstep] += [torch.tensor([]) for _ in range(self.healpix_cells)]
        self.target_times_raw[fstep] += [
            np.array([], dtype="datetime64[ns]") for _ in range(self.healpix_cells)
        ]

    def add_source(
        self, ss_raw: IOReaderData, ss_lens: torch.tensor, ss_cells: list, ss_centroids: list
    ) -> None:
        """
        Add data for source for one input.

        Parameters
        ----------
        ss_raw : IOReaderData( dataclass containing coords, geoinfos, data, and datetimes )
        ss_lens : torch.tensor( number of healpix cells )
        ss_cells : list( number of healpix cells )
            [ torch.tensor( tokens per cell, token size, number of channels) ]
        ss_centroids : list(number of healpix cells )
            [ torch.tensor( for source , 5) ]

        Returns
        -------
        None
        """

        self.source_raw = ss_raw
        self.source_tokens_lens = ss_lens
        self.source_tokens_cells = torch.cat(ss_cells)
        self.source_centroids = torch.cat(ss_centroids)

        idx = torch.isnan(self.source_tokens_cells)
        self.source_tokens_cells[idx] = self.mask_value

    def add_target(
        self,
        fstep: int,
        targets: list,
        target_coords: torch.tensor,
        target_coords_raw: torch.tensor,
        times_raw: torch.tensor,
    ) -> None:
        """
        Add data for target for one input.

        Parameters
        ----------
        fstep : int
            forecast step
        targets : torch.tensor( number of healpix cells )
            [ torch.tensor( num tokens, channels) ]
              Target data for loss computation
        targets_lens : torch.tensor( number of healpix cells)
            length of targets per cell
        target_coords : list( number of healpix cells)
            [ torch.tensor( points per cell, 105) ]
              target coordinates
        target_times : list( number of healpix cells)
            [ torch.tensor( points per cell) ]
              absolute target times

        Returns
        -------
        None
        """

        self.target_tokens[fstep] = torch.cat(targets)
        self.target_coords[fstep] = torch.cat(target_coords)
        self.target_times_raw[fstep] = np.concatenate(times_raw)
        self.target_coords_raw[fstep] = torch.cat(target_coords_raw)

        tc = target_coords
        self.target_coords_lens[fstep] = torch.tensor(
            [len(f) for f in tc] if len(tc) > 1 else self.target_coords_lens[fstep],
            dtype=torch.int,
        )
        self.target_tokens_lens[fstep] = torch.tensor(
            [len(f) for f in targets] if len(targets) > 1 else self.target_tokens_lens[fstep],
            dtype=torch.int,
        )

    def target_empty(self) -> bool:
        """
        Test if target for stream is empty

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True if target is empty for stream, else False
        """

        # cat over forecast steps
        return torch.cat(self.target_tokens_lens).sum() == 0

    def source_empty(self) -> bool:
        """
        Test if source for stream is empty

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True if target is empty for stream, else False
        """

        return self.source_tokens_lens.sum() == 0

    def empty(self):
        """
        Test if stream (source and target) are empty

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True if stream is empty for stream, else False
        """

        return self.source_empty() and self.target_empty()

    def is_spoof(self) -> bool:
        """
        Either source or target is spoof
        """
        return self.source_is_spoof or self.target_is_spoof


def spoof(healpix_level: int, datetime, geoinfo_size, mean_of_data) -> IOReaderData:
    """
    Spoof an instance from data_reader_base.ReaderData instance.
    other should be such an instance.
    """

    dx = 0.5
    dy = 0.5
    num_healpix_cells = 12 * 4**healpix_level
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**healpix_level, dx=dx, dy=dy, order="nested"
    )
    coords = np.stack([lats.deg, lons.deg], axis=-1, dtype=np.float32)
    geoinfos = np.zeros((coords.shape[0], geoinfo_size), dtype=np.float32)

    data = np.expand_dims(mean_of_data.astype(np.float32), axis=0).repeat(coords.shape[0], axis=0)
    datetimes = np.array(datetime).repeat(coords.shape[0])

    n_datapoints = len(data)

    assert coords.shape == (n_datapoints, 2), (
        "number of datapoints do not match data",
        coords.shape,
        (n_datapoints, 2),
    )
    assert geoinfos.shape[0] == n_datapoints, (
        "number of datapoints do not match data",
        geoinfos.shape,
        (n_datapoints, geoinfo_size),
    )
    assert datetimes.shape[0] == n_datapoints, (
        "number of datapoints do not match data",
        datetimes.shape,
        (n_datapoints,),
    )

    return IOReaderData(coords, geoinfos, data, datetimes)
