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

    def __init__(self, idx: int, input_steps: int, forecast_steps: int, healpix_cells: int) -> None:
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

        self.input_steps = input_steps
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
        self.idxs_inv = [torch.tensor([], dtype=torch.int64) for _ in range(forecast_steps + 1)]

        # source tokens per cell
        self.source_tokens_cells = [None for _ in range(self.input_steps)]
        # length of source tokens per cell (without padding)
        self.source_tokens_lens = [[] for _ in range(self.input_steps)]
        # unprocessed source (for logging)
        self.source_raw = [None for _ in range(self.input_steps)]
        # auxiliary data for scatter operation that changes from stream-centric to cell-centric
        # processing after embedding
        self.source_idxs_embed = [None for _ in range(self.input_steps)]
        self.source_idxs_embed_pe = [None for _ in range(self.input_steps)]

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

        dv = device
        self.target_coords = [t.to(dv, non_blocking=True) for t in self.target_coords]
        self.target_tokens = [t.to(dv, non_blocking=True) for t in self.target_tokens]

        # move to device if source data is present
        if not np.array([s is None for s in self.source_tokens_cells]).all():
            self.source_tokens_cells = [
                s.to(dv, non_blocking=True) for s in self.source_tokens_cells
            ]
            self.source_tokens_lens = [s.to(dv, non_blocking=True) for s in self.source_tokens_lens]

            self.source_idxs_embed = [s.to(dv, non_blocking=True) for s in self.source_idxs_embed]
            self.source_idxs_embed_pe = [
                s.to(dv, non_blocking=True) for s in self.source_idxs_embed_pe
            ]

        return self

    def add_source(
        self, step: int, ss_raw: IOReaderData, ss_lens: torch.Tensor, ss_cells: list
    ) -> None:
        """
        Add data for source for one input.

        Parameters
        ----------
        ss_raw : IOReaderData( dataclass containing coords, geoinfos, data, and datetimes )
        ss_lens : torch.Tensor( number of healpix cells )
        ss_cells : list( number of healpix cells )
            [ torch.Tensor( tokens per cell, token size, number of channels) ]

        Returns
        -------
        None
        """

        assert step < self.input_steps

        self.source_raw[step] = ss_raw
        self.source_tokens_lens[step] = ss_lens
        self.source_tokens_cells[step] = torch.stack(ss_cells)

        idx = torch.isnan(self.source_tokens_cells[step])
        self.source_tokens_cells[step][idx] = self.mask_value

    def add_target(
        self,
        fstep: int,
        targets: list,
        target_coords: torch.Tensor,
        target_coords_per_cell: torch.Tensor,
        target_coords_raw: torch.Tensor,
        times_raw: torch.Tensor,
        idxs_inv: torch.Tensor,
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
        idxs_inv:
            Indices to reorder targets back to order in input

        Returns
        -------
        None
        """

        self.target_tokens[fstep] = targets
        self.target_coords[fstep] = target_coords
        self.target_coords_lens[fstep] = target_coords_per_cell
        self.target_times_raw[fstep] = times_raw
        self.target_coords_raw[fstep] = target_coords_raw
        self.idxs_inv[fstep] = idxs_inv

    def add_target_values(
        self,
        fstep: int,
        targets: list,
        target_coords_raw: torch.Tensor,
        times_raw: torch.Tensor,
        idxs_inv: torch.Tensor,
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
        idxs_inv:
            Indices to reorder targets back to order in input

        Returns
        -------
        None
        """

        self.target_tokens[fstep] = targets
        self.target_times_raw[fstep] = times_raw
        self.target_coords_raw[fstep] = target_coords_raw
        self.idxs_inv[fstep] = idxs_inv

    def add_target_coords(
        self,
        fstep: int,
        target_coords: torch.Tensor,
        target_coords_per_cell: torch.Tensor,
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
        idxs_inv:
            Indices to reorder targets back to order in input

        Returns
        -------
        None
        """

        self.target_coords[fstep] = target_coords
        self.target_coords_lens[fstep] = target_coords_per_cell

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
        target_coords_empty = torch.cat(self.target_coords_lens).sum() == 0
        target_tokens_empty = torch.cat(self.target_tokens).sum() == 0
        return target_coords_empty and target_tokens_empty

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

        return (
            torch.tensor([s.sum() if len(s) > 0 else 0 for s in self.source_tokens_lens]).sum() == 0
        )

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

    def get_forecast_steps(self) -> int:
        """
        Get number of forecast steps
        """
        return self.forecast_steps


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
