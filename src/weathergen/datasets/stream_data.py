# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.train.utils import TRAIN, Stage


def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Pin a tensor to CPU pinned memory.

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    torch.Tensor
        The pinned tensor.
    """
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


def _pin_tensor_list(tensor_list: list) -> list:
    """Pin all tensors in a list to CPU pinned memory.

    Parameters
    ----------
    tensor_list : list
        List of tensors (or other objects) to pin.

    Returns
    -------
    list
        List with all torch.Tensor elements pinned to CPU pinned memory.
    """
    return [_pin_tensor(t) for t in tensor_list]


@dataclass
class DecoderTargets:
    """Decoder-cell ordered coordinates and indices into the native masked target rows.

    Lengths contains one count per compact decoder cell, without a leading zero.
    """

    coords: torch.Tensor
    lengths: torch.Tensor
    indices: torch.Tensor

    @classmethod
    def empty(cls, num_cells: int, coord_dim: int):
        return cls(
            torch.empty((0, coord_dim)),
            torch.zeros(num_cells, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
        )

    def pin_memory(self):
        self.coords = _pin_tensor(self.coords)
        self.lengths = _pin_tensor(self.lengths)
        self.indices = _pin_tensor(self.indices)
        return self

    def to_device(self, device):
        self.coords = self.coords.to(device, non_blocking=True)
        self.lengths = self.lengths.to(device, non_blocking=True)
        self.indices = self.indices.to(device, non_blocking=True)
        return self


class StreamData:
    """
    StreamData object that encapsulates all data the model ingests for one batch item
    for one stream.
    """

    def __init__(
        self,
        idx: int,
        input_steps: int,
        output_steps: int,
        healpix_cells: int,
    ) -> None:
        """
        StreamData object

        Parameters
        ----------
        idx: int
            Time index for this sample
        input_steps : int
            Number of input steps
        output_steps : int
            Number of output steps
            Note -- Last input step and first output step always overlap.
        healpix_cells : int
            Number of healpix cells for source

        Returns
        -------
        None
        """

        self.mask_value = 0.0

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.healpix_cells = healpix_cells

        self.source_is_spoof = [False for _ in range(self.input_steps)]
        self.target_is_spoof = [False for _ in range(self.output_steps)]

        # initialize empty members
        self.sample_idx = idx
        self.target_coords_raw = [[] for _ in range(output_steps)]
        self.target_times_raw = [np.array([], dtype="datetime64[ns]") for _ in range(output_steps)]
        self.target_tokens = [torch.tensor([]) for _ in range(output_steps)]
        self.idxs_inv = [torch.tensor([], dtype=torch.int64) for _ in range(output_steps)]
        self.decoder_targets: dict[str, list[DecoderTargets]] = {}

        # source tokens per cell
        self.source_tokens_cells = [None for _ in range(self.input_steps)]
        # length of source tokens per cell (without padding)
        self.source_tokens_lens = [
            torch.zeros(self.healpix_cells, dtype=torch.int32) for _ in range(self.input_steps)
        ]
        # unprocessed source (for logging)
        self.source_raw = [None for _ in range(self.input_steps)]
        # auxiliary data for scatter operation that changes from stream-centric to cell-centric
        # processing after embedding
        self.source_idxs_embed = [torch.tensor([]) for _ in range(self.input_steps)]
        self.source_idxs_embed_pe = [torch.tensor([]) for _ in range(self.input_steps)]

    def pin_memory(self):
        """Pin all tensors in this StreamData object to CPU pinned memory"""

        # Pin target tensors
        self.target_tokens = _pin_tensor_list(self.target_tokens)
        self.idxs_inv = _pin_tensor_list(self.idxs_inv)
        self.target_coords_raw = _pin_tensor_list(self.target_coords_raw)
        for targets in self.decoder_targets.values():
            for target in targets:
                target.pin_memory()

        # Pin source tensors
        self.source_tokens_cells = _pin_tensor_list(self.source_tokens_cells)
        self.source_tokens_lens = _pin_tensor_list(self.source_tokens_lens)
        self.source_idxs_embed = _pin_tensor_list(self.source_idxs_embed)

        return self

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
        self.target_tokens = [t.to(dv, non_blocking=True) for t in self.target_tokens]
        for targets in self.decoder_targets.values():
            for target in targets:
                target.to_device(dv)

        # move to device if source data is present
        if not np.array([s is None for s in self.source_tokens_cells]).all():
            self.source_tokens_cells = [
                s.to(dv, non_blocking=True) for s in self.source_tokens_cells
            ]
            self.source_tokens_lens = [s.to(dv, non_blocking=True) for s in self.source_tokens_lens]

            self.source_idxs_embed = [s.to(dv, non_blocking=True) for s in self.source_idxs_embed]

        return self

    def add_source(
        self,
        stage: Stage,
        step: int,
        ss_raw: IOReaderData,
        ss_lens: torch.Tensor,
        ss_cells: list,
        is_spoof: bool,
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

        if stage == TRAIN:
            del ss_raw
            ss_raw = None

        self.source_raw[step] = ss_raw
        self.source_tokens_lens[step] = ss_lens
        self.source_tokens_cells[step] = torch.stack(ss_cells)

        idx = torch.isnan(self.source_tokens_cells[step])
        self.source_tokens_cells[step][idx] = self.mask_value

        self.source_is_spoof[step] = is_spoof

    def add_target_values(
        self,
        stage: Stage,
        fstep: int,
        targets: list,
        target_coords_raw: torch.Tensor,
        times_raw: torch.Tensor,
        idxs_inv: torch.Tensor,
        is_spoof: bool,
    ) -> None:
        """
        Add data for target for one input.

        Parameters
        ----------
        fstep : int
            forecast step
        targets : torch.Tensor
            Masked native target values, grouped by native cell.
        target_coords_raw : torch.Tensor
            Raw latitude/longitude for the same target rows.
        times_raw : numpy.ndarray
            Absolute times for the same target rows.
        idxs_inv:
            Indices to reorder targets back to order in input

        Returns
        -------
        None
        """

        if stage == TRAIN:
            del idxs_inv
            idxs_inv = None

        self.target_tokens[fstep] = targets
        self.target_times_raw[fstep] = times_raw
        self.target_coords_raw[fstep] = target_coords_raw
        self.idxs_inv[fstep] = idxs_inv

        self.target_is_spoof[fstep] = is_spoof

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

        coords_empty = all(
            target.lengths.sum() == 0
            for targets in self.decoder_targets.values()
            for target in targets
        )
        return coords_empty and all(target.numel() == 0 for target in self.target_tokens)

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

    def target_nan(self) -> bool:
        """
        Check if target for stream is all NaN

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True if target is empty for stream, else False
        """

        is_nan = torch.isnan(torch.cat(self.target_tokens))
        return is_nan.all() if len(is_nan) > 0 else False

    def source_nan(self) -> bool:
        """
        Check if source for stream is all NaN

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True if source is all NaN for stream, else False
        """

        is_nan = torch.tensor(
            [
                torch.isnan(s.coords).all() or torch.isnan(s.data).all()
                for s in self.source_raw
                if s is not None
            ]
        )
        return is_nan.all() if len(is_nan) > 0 else False

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

    def nan(self) -> bool:
        """
        Check if stream (source and target) are all NaN

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True if stream is all NaN
        """

        return self.source_nan() and self.target_nan()

    def is_spoof(self, step: int) -> bool:
        """
        Either source or target at step is spoof
        """
        return any(self.source_is_spoof) or self.target_is_spoof[step]

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps
        """
        return len(self.source_tokens_cells)

    def get_num_target_steps(self) -> int:
        """
        Get number of target steps
        """
        return len(self.target_tokens)


def spoof(healpix_level: int, datetime, geoinfo_size, num_channels, domain=None) -> IOReaderData:
    """
    Spoof an instance from data_reader_base.ReaderData instance.
    other should be such an instance.
    """

    # Generate only the two sampled cells, within the configured domain.
    cells = domain.active_cells if domain is not None else np.arange(12 * 4**healpix_level)
    selected = np.random.choice(cells, size=2, replace=len(cells) < 2)
    lons, lats = hp.healpix_to_lonlat(selected, 2**healpix_level, order="nested")
    coords = np.stack([lats.deg, lons.deg - 180.0], axis=-1, dtype=np.float32)

    geoinfos = np.zeros((coords.shape[0], geoinfo_size), dtype=np.float32)
    data = np.zeros((coords.shape[0], num_channels), dtype=np.float32)
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
