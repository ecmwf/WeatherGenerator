# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch


class StreamData:
    """
    StreamData object that encapsulates all data the model ingests for one batch item
    for one stream.
    """

    def __init__(self, forecast_steps: int, nhc_source: int, nhc_target: int) -> None:
        """
        Create StreamData object.

        Parameters
        ----------
        forecast_steps : int
            Number of forecast steps
        nhc_source : int
            Number of healpix cells for source
        nhc_target : int
            Number of healpix cells for target

        Returns
        -------
        None
        """

        self.mask_value = 0.0

        self.forecast_steps = forecast_steps
        self.nhc_source = nhc_source
        self.nhc_target = nhc_target

        # TODO add shape of tensors

        # initialize empty members
        self.target_coords = [[] for _ in range(forecast_steps + 1)]
        self.target_coords_raw = [[] for _ in range(forecast_steps + 1)]
        self.target_times_raw = [[] for _ in range(forecast_steps + 1)]
        # this is not directly used but to precompute index in compute_idxs_predict()
        self.target_coords_lens = [[] for _ in range(forecast_steps + 1)]
        self.target_tokens = [[] for _ in range(forecast_steps + 1)]
        self.target_tokens_lens = [[] for _ in range(forecast_steps + 1)]
        # source tokens per cell
        self.source_tokens_cells = []
        # length of source tokens per cell (without padding)
        self.source_tokens_lens = []
        self.source_centroids = []
        # unaltered source (for logging)
        self.source_raw = []
        # auxiliary data for scatter operation that changes from stream-centric to cell-centric
        # processing after embedding
        self.source_idxs_embed = torch.tensor([])
        self.source_idxs_embed_pe = torch.tensor([])

    def to_device(self, device="cuda") -> None:
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

        return self

    def add_empty_source(self) -> None:
        """
        Add an empty source for an input.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.source_raw += [torch.tensor([])]
        self.source_tokens_lens += [torch.zeros([self.nhc_source], dtype=torch.int32)]
        self.source_tokens_cells += [torch.tensor([])]
        self.source_centroids += [torch.tensor([])]

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

        self.target_tokens[fstep] += [torch.tensor([])]
        self.target_tokens_lens[fstep] += [torch.zeros([self.nhc_target], dtype=torch.int32)]
        self.target_coords[fstep] += [torch.tensor([])]
        self.target_coords_lens[fstep] += [torch.zeros([self.nhc_target], dtype=torch.int32)]
        self.target_coords_raw[fstep] += [torch.tensor([])]
        self.target_times_raw_raw[fstep] += [np.array([], dtype="datetime64[ns]")]

    def add_source(
        self, ss_raw: torch.tensor, ss_lens: torch.tensor, ss_cells: list, ss_centroids: list
    ) -> None:
        """
        Add data for source for one input.

        Parameters
        ----------
        ss_raw : torch.tensor( number of data points in time window , number of channels )
        ss_lens : torch.tensor( number of healpix cells )
        ss_cells : list( number of healpix cells )[ torch.tensor( tokens per cell, token size, number of channels) ]
        ss_centroids : list(number of healpix cells )[ torch.tensor( for source , 5) ]

        Returns
        -------
        None
        """

        self.source_raw += [ss_raw]
        self.source_tokens_lens += [ss_lens]
        self.source_tokens_cells += [ss_cells]
        self.source_centroids += [ss_centroids]

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
        targets : torch.tensor( number of healpix cells )[ torch.tensor( num tokens, channels) ]
            Target data for loss computation
        targets_lens : torch.tensor( number of healpix cells)
            length of targets per cell
        target_coords : list( number of healpix cells)[ torch.tensor( points per cell, 105)]
            target coordinates
        target_times : list( number of healpix cells)[ torch.tensor( points per cell)]
            absolute target times

        Returns
        -------
        None
        """

        self.target_tokens[fstep] += [targets]
        self.target_coords[fstep] += [target_coords]
        self.target_coords_raw[fstep] += [target_coords_raw]
        self.target_times_raw[fstep] += [times_raw]

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

    ####################################################################################################
    def _merge_cells(self, s_list: list, num_healpix_cells: int) -> list:
        """
        Helper function to merge different inputs for the stream
        (preserving in particular the per cell information)

        Parameters
        ----------
        s_list : list( number of healpix cells)[torch.tensor]
            List of lists to be merged along first (multi-source) dimension
        num_healpix_cells : int
            Number of healpix cells (equal to second dimension of list for all list items)

        Returns
        -------
        list
            Merged inputs
        """

        if torch.tensor([len(s) for s in s_list]).sum() == 0:
            return torch.tensor([]) if type(s_list[0][0]) is torch.Tensor else np.array([])

        cat = torch.cat if type(s_list[0][0]) is torch.Tensor else np.concatenate
        ret = cat(
            [
                cat([s_list[i_s][i] for i_s in range(len(s_list)) if len(s_list[i_s]) > 0])
                for i in range(num_healpix_cells)
            ]
        )

        return ret

    def merge_inputs(self) -> None:
        """
        Merge sources and targets from different inputs for the stream
        (preserving in particular the per cell information)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # collect all sources in current stream and add to batch sample list when non-empty
        if torch.tensor([len(s) for s in self.source_tokens_cells]).sum() > 0:
            self.source_raw = torch.cat(self.source_raw)

            # collect by merging entries per cells, preserving cell structure
            self.source_tokens_cells = self._merge_cells(self.source_tokens_cells, self.nhc_source)
            self.source_centroids = self._merge_cells(self.source_centroids, self.nhc_source)
            # lens can be stacked and summed
            self.source_tokens_lens = torch.stack(self.source_tokens_lens).sum(0)

            # remove NaNs
            idx = torch.isnan(self.source_tokens_cells)
            self.source_tokens_cells[idx] = self.mask_value
            idx = torch.isnan(self.source_centroids)
            self.source_centroids[idx] = self.mask_value

        else:
            self.source_raw = torch.tensor([])
            self.source_tokens_lens = torch.zeros([self.nhc_source])
            self.source_tokens_cells = torch.tensor([])
            self.source_centroids = torch.tensor([])

        # targets
        for fstep in range(len(self.target_coords)):
            # collect all targets in current stream and add to batch sample list when non-empty
            if torch.tensor([len(s) for s in self.target_tokens[fstep]]).sum() > 0:
                nt = self.nhc_target

                self.target_coords_lens[fstep] = torch.tensor(
                    [[len(f) for f in ff] for ff in self.target_coords[fstep]]
                ).sum(0)
                self.target_tokens_lens[fstep] = torch.tensor(
                    [[len(f) for f in ff] for ff in self.target_tokens[fstep]]
                ).sum(0)
                self.target_coords[fstep] = self._merge_cells(self.target_coords[fstep], nt)
                self.target_coords_raw[fstep] = self._merge_cells(self.target_coords_raw[fstep], nt)

                self.target_times_raw[fstep] = self._merge_cells(self.target_times_raw[fstep], nt)
                self.target_tokens[fstep] = self._merge_cells(self.target_tokens[fstep], nt)
                # remove NaNs
                # TODO: it seems better to drop data points with NaN values in the coords than
                #       to mask them
                assert not torch.isnan(self.target_coords[fstep]).any()
                # self.target_coords[fstep][torch.isnan(self.target_coords[fstep])] = self.mask_value

            else:
                # TODO: is this branch still needed
                self.target_coords[fstep] = torch.tensor([])
                self.target_coords_raw[fstep] = torch.tensor([])
                self.target_times_raw[fstep] = np.array([], dtype="datetime64[ns]")
                self.target_tokens[fstep] = torch.tensor([])
                self.target_tokens_lens[fstep] = torch.tensor([])
                self.target_coords_lens[fstep] = torch.tensor([])
