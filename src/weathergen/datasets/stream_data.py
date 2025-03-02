# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import torch

from weathergen.datasets.utils import merge_cells


class StreamData() :
    """
        StreamData object that encapsulates all data the model ingests for the stream 
        for a batch item.
    """

    def __init__( self, forecast_steps : int, nhc_source : int, nhc_target : int) -> None :
        """
            Create StreamData object.

            Parameters
            ----------
            forecast_steps : int
                Number of forecast steps

            Returns
            -------
            StreamData
                The created StreamData.
        """

        self.mask_value = 0.

        self.forecast_steps = forecast_steps
        self.nhc_source = nhc_source
        self.nhc_target = nhc_target

        # initialize empty members 
        self.target_coords = [[] for _ in range(forecast_steps + 1)]
        self.target_coords_lens = [[] for _ in range(forecast_steps + 1)]
        self.target_tokens = [[] for _ in range(forecast_steps + 1)]
        self.target_tokens_lens = [[] for _ in range(forecast_steps + 1)]
        # source tokens per cell
        self.source_tokens_cells = []
        # length of source tokens per cell (without padding)
        self.source_tokens_lens = []
        self.source_centroids = []
        # 
        self.source_raw = []
        #
        self.source_idxs_embed = torch.tensor([])
        self.source_idxs_embed_pe = torch.tensor([])

    def to_device( self, device = 'cuda') -> None :
        """
            Move data to GPU
        """

        self.source_tokens_cells = self.source_tokens_cells.to( device, non_blocking=True)
        self.source_centroids = self.source_centroids.to( device, non_blocking=True)
        self.source_tokens_lens = self.source_tokens_lens.to( device, non_blocking=True)

        self.target_coords = [t.to( device, non_blocking=True) for t in self.target_coords]
        self.target_coords_lens = [t.to( device, non_blocking=True) for t in self.target_coords_lens]
        self.target_tokens = [t.to( device, non_blocking=True) for t in self.target_tokens]
        self.target_tokens_lens = [t.to( device, non_blocking=True) for t in self.target_tokens_lens]

        self.source_idxs_embed = self.source_idxs_embed.to( device, non_blocking=True)
        self.source_idxs_embed_pe = self.source_idxs_embed_pe.to( device, non_blocking=True)

        return self

    def add_empty_source( self) -> None :
        """
            
        """

        self.source_raw += [ torch.tensor([]) ]
        self.source_tokens_lens += [ torch.zeros([self.nhc_source], dtype=torch.int32) ]
        self.source_tokens_cells += [ torch.tensor([]) ]
        self.source_centroids += [ torch.tensor([]) ]
    
    def add_empty_target( self, 
                          fstep : int) -> None :
        """
            
        """

        self.target_tokens_lens[fstep] += [ torch.zeros([self.nhc_target], dtype=torch.int32) ]
        self.target_tokens[fstep] += [torch.tensor([]) ]
        self.target_coords[fstep] += [ torch.tensor([]) ]
        self.target_coords_lens[fstep] += [ torch.zeros([self.nhc_target], dtype=torch.int32) ]

    def add_source( self, 
                    source1_raw : torch.tensor, 
                    ss_lens : torch.tensor,
                    ss_cells : torch.tensor,
                    ss_centroids : torch.tensor
                    ) -> None :
        """

        """

        self.source_raw += [source1_raw]
        self.source_tokens_lens += [ss_lens]
        self.source_tokens_cells += [ss_cells]
        # TODO: is the if clauses still needed?
        self.source_centroids += (
            [ss_centroids] if len(ss_centroids) > 0 else [torch.tensor([])]
        )

    def add_target( self, 
                    fstep : int, 
                    tt_cells : torch.tensor,
                    tt_lens : torch.tensor, 
                    tc : torch.tensor,
                    tc_lens : torch.tensor
                    ) -> None :
        """

        """

            # TODO: are the if clauses still needed?
        self.target_tokens_lens[fstep] += (
            [tt_lens] if len(tt_lens) > 0 else [torch.tensor([])]
        )
        self.target_tokens[fstep] += (
            [tt_cells] if len(tt_cells) > 0 else [torch.tensor([])]
        )
        self.target_coords[fstep] += [tc]
        self.target_coords_lens[fstep] += [tc_lens]

    def target_empty( self) :
        """

        """

        # cat over forecast steps
        return torch.cat(self.target_tokens_lens).sum() == 0

    def source_empty( self) :
        """

        """

        return self.source_tokens_lens.sum() == 0

    def empty( self) :
        """

        """

        return self.source_empty() and self.target_empty()

    def merge_inputs( self) -> None :
        """
        """

        # collect all sources in current stream and add to batch sample list when non-empty
        if torch.tensor([len(s) for s in self.source_tokens_cells]).sum() > 0:
            
            self.source_raw = torch.cat(self.source_raw)

            # collect by merging entries per cells, preserving cell structure
            self.source_tokens_cells = merge_cells( self.source_tokens_cells, self.nhc_source)
            self.source_centroids = merge_cells( self.source_centroids, self.nhc_source)
            # lens can be stacked and summed
            self.source_tokens_lens = torch.stack( self.source_tokens_lens).sum(0)
            
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
        for fstep in range( len(self.target_coords)):
            # collect all targets in current stream and add to batch sample list when non-empty
            if torch.tensor([len(s) for s in self.target_tokens[fstep]]).sum() > 0:
                self.target_coords[fstep] = merge_cells(self.target_coords[fstep], self.nhc_target)
                self.target_tokens[fstep] = merge_cells(self.target_tokens[fstep], self.nhc_target)
                # lens can be stacked and summed
                self.target_tokens_lens[fstep] = torch.stack(self.target_tokens_lens[fstep]).sum(0)
                self.target_coords_lens[fstep] = torch.stack(self.target_coords_lens[fstep]).sum(0)
                # remove NaNs
                self.target_coords[fstep][torch.isnan(self.target_coords[fstep])] = self.mask_value

            else:
                # TODO: is this branch still needed
                self.target_coords[fstep] = torch.tensor([])
                self.target_coords_lens[fstep] = torch.tensor([])
                self.target_tokens[fstep] = torch.tensor([])
                self.target_tokens_lens[fstep] = torch.tensor([])
