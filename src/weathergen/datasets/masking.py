import logging

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData

_logger = logging.getLogger(__name__)


# Convert to torch.bool
def to_bool_tensor(arr):
    return torch.from_numpy(np.asarray(arr)).to(torch.bool)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "block", "healpix", "cropping_healpix", "channel").
        current_strategy (str): The current strategy in use, relevant
                                when using "combination" strategy.
        "random" - random masking of tokens at the level of the data
        "block" - masking out large blocks of tokens in 1D, without spatial meaning
        "healpix" - masking at the level of HEALPix cells, where all child cells
                    of a parent cell at a specific HEALpix level are masked
                    if the parent is masked.
                    The healpix level must be configured with hl_mask.
                    e.g. masking_strategy_config = {"hl_mask": 1}
                    with hl_mask the level for masking that we want to apply
                    e.g. level 1 very large cells masked
        "cropping_healpix" - spatial cropping that keeps spatially contiguous regions
                    and masks everything else. Uses neighbor relationships or geodesic
                    distance to ensure spatial contiguity. Perfect for DINO/JEPA/IBOT.
                    e.g. masking_strategy_config = {"hl_mask": 0, "method": "geodesic_disk"}
                    method: "disk" (neighbor growth), "random_walk", or "geodesic_disk" (circular)
        "channel" - masking data channels, where channels of the data are masked
                    can be done per-cell (each cell has different channels masked)
                    or globally (all have the same channels masked).
                    e.g. masking_strategy_config = {"mode": "per_cell"} or
                    {"mode": "global"}
        "causal" - masking the latest timesteps in each token, according to the masking rate.
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", etc.
                                        specific to the masking strategy. See above.
    """

    def __init__(self, cf: Config):
        self.rng = None
        # self.masking_rate = cf.masking_rate
        # self.masking_strategy = cf.masking_strategy
        # self.current_strategy = cf.masking_strategy  # Current strategy in use
        # self.masking_rate_sampling = cf.masking_rate_sampling
        # # masking_strategy_config is a dictionary that can hold any additional parameters
        # self.masking_strategy_config = cf.get("masking_strategy_config", {})
        # self.perm_sel = None
        # self.mask_tokens = None
        # self.mask_channels = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_level_data = cf.healpix_level
        self.healpix_num_cells = 12 * (4**cf.healpix_level)

    # # Per-batch strategy tracking
    # self.same_strategy_per_batch = self.masking_strategy_config.get(
    #     "same_strategy_per_batch", False
    # )
    # self.batch_strategy_set = False

    # # Check for required masking_strategy_config at construction time
    # if self.current_strategy == "healpix":
    #     hl_data = self.healpix_level_data
    #     hl_mask = self.masking_strategy_config.get("hl_mask")
    #     assert hl_data is not None and hl_mask is not None, (
    #         "If HEALPix masking, hl_mask must be given in masking_strategy_config."
    #     )
    #     assert hl_mask < hl_data, "hl_mask must be less than hl_data for HEALPix masking."

    # if self.current_strategy == "channel":
    #     # Ensure that masking_strategy_config contains either 'global' or 'per_cell'
    #     assert self.masking_strategy_config.get("mode") in [
    #         "global",
    #         "per_cell",
    #     ], "masking_strategy_config must contain 'mode' key with value 'global' or 'per_cell'."

    #     # check all streams that source and target channels are identical
    #     for stream in cf.streams:
    #         # check explicit includes
    #         source_include = stream.get("source_include", [])
    #         target_include = stream.get("target_include", [])
    #         assert set(source_include) == set(target_include), (
    #             "Source and target channels not identical. Required for masking_mode=channel"
    #         )
    #         # check excludes
    #         source_exclude = stream.get("source_exclude", [])
    #         target_exclude = stream.get("target_exclude", [])
    #         assert set(source_exclude) == set(target_exclude), (
    #             "Source and target channels not identical. Required for masking_mode=channel"
    #         )

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.rng = rng

    def _select_spatially_contiguous_cells(
        self,
        healpix_level: int,
        num_cells_to_select: int,
        center_cell: int | None = None,
        method: str = "disk",
        overlap_with: np.ndarray | None = None,
        overlap_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Select spatially contiguous cells on the sphere using neighbor relationships.

        This is the core spatial selection helper used for both masking and cropping.

        Args:
            healpix_level: HEALPix level for selection
            num_cells_to_select: Number of cells to select
            center_cell: Starting cell (None = random, or optimized for overlap if specified)
            method: Selection method:
                - "disk": Layer-by-layer neighbor growth (compact regions)
                - "random_walk": Random neighbor selection (irregular shapes)
                - "geodesic_disk": Angular distance selection (circular regions, best for SSL)
            overlap_with: Existing crop to control overlap with (for IBOT-style training)
            overlap_ratio: Target overlap ratio [0.0-1.0] (requires overlap_with)
                         0.0 = no overlap, 0.5 = 50% overlap, 1.0 = complete overlap

        Returns:
            Array of selected cell indices forming a spatially contiguous region

        Examples:
            # Independent crop
            crop1 = _select_spatially_contiguous_cells(0, 9, method="geodesic_disk")

            # Crop with 30% overlap (IBOT-style)
            crop2 = _select_spatially_contiguous_cells(0, 9, method="geodesic_disk",
                                                       overlap_with=crop1, overlap_ratio=0.3)
        """
        import warnings
        import astropy_healpix as hp

        num_total_cells = 12 * (4**healpix_level)
        nside = 2**healpix_level

        assert num_cells_to_select <= num_total_cells

        # Optimize center for controlled overlap if requested
        # NOTE: This is the programmatic API approach. The config system uses a more
        # deterministic approach in _generate_cell_mask (see lines 752-803).
        if overlap_with is not None and overlap_ratio is not None and center_cell is None:
            assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio must be in [0.0, 1.0]"

            overlap_set = set(overlap_with)

            # Use intelligent center selection based on overlap target
            if overlap_ratio > 0.7:
                # High overlap: select center from within existing crop
                center_cell = self.rng.choice(list(overlap_set))
            elif overlap_ratio < 0.3:
                # Low overlap: select center from outside existing crop
                non_overlap_cells = [c for c in range(num_total_cells) if c not in overlap_set]
                if non_overlap_cells:
                    center_cell = self.rng.choice(non_overlap_cells)
                else:
                    # Fallback if no non-overlap cells available
                    center_cell = self.rng.integers(0, num_total_cells)
            else:
                # Medium overlap: random selection (boundary-agnostic)
                center_cell = self.rng.integers(0, num_total_cells)

        # Random starting point if not specified
        elif center_cell is None:
            center_cell = self.rng.integers(0, num_total_cells)

        if method == "disk":
            # Layer-by-layer neighbor growth - creates compact irregular regions
            selected = {center_cell}
            frontier = {center_cell}

            while len(selected) < num_cells_to_select and frontier:
                # Expand frontier by one layer
                next_frontier = set()
                for cell in frontier:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered")
                        neighbors = hp.neighbours(cell, nside, order="nested")
                    valid_neighbors = [n for n in neighbors if n != -1 and n not in selected]
                    next_frontier.update(valid_neighbors)

                if not next_frontier:
                    break

                # Randomly select from frontier to reach target count
                candidates = list(next_frontier)
                self.rng.shuffle(candidates)
                num_to_add = min(len(candidates), num_cells_to_select - len(selected))
                selected.update(candidates[:num_to_add])
                frontier = set(candidates[:num_to_add])

        elif method == "random_walk":
            # Random walk through neighbors - creates elongated irregular regions
            selected = {center_cell}
            frontier = {center_cell}

            while len(selected) < num_cells_to_select:
                # Get all neighbors of current frontier
                neighbors = set()
                for cell in frontier:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered")
                        cell_neighbors = hp.neighbours(cell, nside, order="nested")
                    valid = [n for n in cell_neighbors if n != -1 and n not in selected]
                    neighbors.update(valid)

                if not neighbors:
                    break

                # Randomly pick one neighbor and continue from there
                next_cell = self.rng.choice(list(neighbors))
                selected.add(next_cell)
                frontier = {next_cell}

        elif method == "geodesic_disk":
            # Angular distance selection - creates most uniform circular regions
            # Best method for SSL (DINO/JEPA/IBOT) due to shape regularity

            def lonlat_to_xyz(lon, lat):
                """Convert lon/lat to 3D cartesian coordinates."""
                return np.array([
                    np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat)
                ])

            # Get center coordinates
            center_lonlat = hp.healpix_to_lonlat(center_cell, nside, order="nested")
            center_lon = float(center_lonlat[0].value if hasattr(center_lonlat[0], 'value') else center_lonlat[0])
            center_lat = float(center_lonlat[1].value if hasattr(center_lonlat[1], 'value') else center_lonlat[1])
            center_xyz = lonlat_to_xyz(center_lon, center_lat)

            # Get all cell coordinates
            all_indices = np.arange(num_total_cells)
            all_lonlat = hp.healpix_to_lonlat(all_indices, nside, order="nested")
            all_lon = all_lonlat[0].value if hasattr(all_lonlat[0], 'value') else all_lonlat[0]
            all_lat = all_lonlat[1].value if hasattr(all_lonlat[1], 'value') else all_lonlat[1]

            all_xyz = np.stack([
                np.cos(all_lat) * np.cos(all_lon),
                np.cos(all_lat) * np.sin(all_lon),
                np.sin(all_lat)
            ], axis=1)

            # Compute angular distances and select closest cells
            dot_products = np.clip(np.dot(all_xyz, center_xyz), -1.0, 1.0)
            angular_distances = np.arccos(dot_products)
            selected = np.argsort(angular_distances)[:num_cells_to_select]

        else:
            raise ValueError(f"Unknown selection method: {method}")

        return np.array(sorted(selected))

    # def set_batch_strategy(self):
    #     """
    #     Set strategy for this batch.
    #     Only relevant with combination and same_strategy_per_batch.
    #     """
    #     if self.masking_strategy == "combination" and self.same_strategy_per_batch:
    #         self.current_strategy = self.rng.choice(
    #             self.masking_strategy_config["strategies"],
    #             p=self.masking_strategy_config["probabilities"],
    #         )
    #         self.batch_strategy_set = True

    # def reset_batch_strategy(self):
    #     """
    #     Reset for next batch.
    #     """
    #     if self.masking_strategy == "combination" and self.same_strategy_per_batch:
    #         self.current_strategy = None
    #         self.batch_strategy_set = False

    def _select_strategy(self):
        """
        Select the strategy to use.
        """
        if self.masking_strategy == "combination":
            if self.same_strategy_per_batch:
                assert self.batch_strategy_set, "Must call set_batch_strategy() first"
                return self.current_strategy
            else:
                # Sample new strategy for each stream
                return self.rng.choice(
                    self.masking_strategy_config["strategies"],
                    p=self.masking_strategy_config["probabilities"],
                )
        else:
            # Non-combination strategy, return as is
            return self.masking_strategy

    def _get_sampling_rate(self):
        """
        Get the sampling, if requested by sampling it itself
        """

        # if masking_rate_sampling is enabled, sample the rate from a normal distribution.
        if self.masking_rate_sampling:
            rate = np.clip(
                np.abs(self.rng.normal(loc=self.masking_rate, scale=1.0 / (2.5 * np.pi))),
                0.01,
                0.99,
            )
        else:
            rate = self.masking_rate

        return rate

    def _generate_healpix_mask(self, token_lens: list[int], rate: float) -> np.typing.NDArray:
        """
        Generates a token-level mask based on hierarchical HEALPix cell selection.

        This method identifies parent cells at a lower resolution (hl_mask) and
        masks all the child cells (and their corresponding tokens) at the data
        resolution (hl_data).

        Args:
            token_lens (list[int]): A list containing the number of tokens in each cell.
            rate (float): The desired masking rate, applied to the parent cells.

        Returns:
            np.ndarray: A flat boolean array (the token-level mask).
        """

        # hl_mask should be provided in masking_strategy_config
        hl_data = self.healpix_level_data
        hl_mask = self.masking_strategy_config.get("hl_mask")

        assert len(token_lens) == self.healpix_num_cells, (
            f"Expected {self.healpix_num_cells} cells at level {hl_data}, got {len(token_lens)}."
        )

        # Calculate the number of parent cells at the mask level (hl_mask)
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff

        rate = self._get_sampling_rate()

        # Choose parent cells to mask based on the specified rate.
        num_parents_to_mask = int(np.round(rate * num_parent_cells))

        if num_parents_to_mask == 0:
            return np.zeros(sum(token_lens), dtype=bool)

        # Select parent cells to mask
        parent_ids_to_mask = self.rng.choice(num_parent_cells, num_parents_to_mask, replace=False)

        # For each parent ID, calculate the child indices and set them in the mask
        parent_ids = np.asarray(parent_ids_to_mask)
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # set mask list for children
        cell_mask = np.zeros(self.healpix_num_cells, dtype=bool)
        cell_mask[child_indices] = True

        # Make the cell-level mask flat and apply it to the token lengths.
        # np.repeat repeats each element of `cell_mask` a number of times specified by `token_lens`.
        flat_mask = np.repeat(cell_mask, token_lens)

        return flat_mask

    def _generate_channel_mask(
        self,
        tokenized_data: list[torch.Tensor],
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[np.typing.NDArray]:
        """
        Generates a channel mask for each cell, handling completely empty tensors.
        This method is robust against cells represented as 1D tensors of shape [0].

        Args:
            tokenized_data (list[torch.Tensor]): A list of tensors. Most will have a shape of
                                                (dim, num_tokens, num_channels), but some may
                                            be empty with a shape of (0,), no data in cell
            rate (float): The desired masking rate for channels.
            coords (torch.Tensor): The coordinates tensor.
            geoinfos (torch.Tensor): The geoinfos tensor.

        Returns:
            list[np.ndarray]: A list of boolean masks. Each mask corresponds to a tensor
                            in tokenized_data.
        """

        if not tokenized_data:
            return []

        # masking rate sampling, to be refactored as shared between methods
        rate = self._get_sampling_rate()

        # isolate the number of actual data channels. 6 refers to time.
        num_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1] + source.shape[-1]
        assert num_channels > 0, "For channel masking, number of channels has to be nonzero."
        num_fixed_channels = self.dim_time_enc + coords.shape[-1] + geoinfos.shape[-1]
        num_data_channels = source.shape[-1]
        mask_count = int(num_data_channels * rate)

        # cat all tokens for efficient processing, split at the end again
        # masks are generated simulatneously for all cells

        tokenized_data_lens = [len(t) for t in tokenized_data]
        tokenized_data_merged = torch.cat(tokenized_data)

        num_tokens = tokenized_data_merged.shape[0]
        token_size = tokenized_data_merged.shape[1]

        if self.masking_strategy_config.get("mode") == "global":
            # generate global mask
            channel_mask = np.zeros(num_channels, dtype=bool)
            m = num_fixed_channels + self.rng.choice(num_data_channels, mask_count, replace=False)
            channel_mask[m] = True

            full_mask = np.zeros_like(tokenized_data_merged).astype(np.bool)
            full_mask[:, :] = channel_mask

        else:  # different mask per cell
            # generate all False mask but with swapped token_size and num_tokens dims so that
            # the masking is constant per token
            channel_mask = np.zeros((token_size, num_tokens, num_channels), dtype=bool)
            # apply masking
            nc = (num_tokens, num_data_channels)
            channel_mask[:, :, num_fixed_channels:] = self.rng.uniform(0, 1, nc) < rate
            # recover correct shape, i.e. swap token_size and num_tokens
            full_mask = channel_mask.transpose([1, 0, 2])

        # split across cells again
        full_mask = np.split(full_mask, np.cumsum(tokenized_data_lens[:-1]))

        return full_mask

    def _generate_causal_mask(
        self,
        tokenized_data: list[torch.Tensor],
        rate: float,
        coords: torch.Tensor,
        geoinfos: torch.Tensor,
        source: torch.Tensor,
    ) -> list[np.typing.NDArray]:
        """
        Generates a causal mask, masking the latest times
        in each tokenized_data according to the masking rate.
        """
        if not tokenized_data:
            return []

        rate = self._get_sampling_rate()

        # Extract all lengths at once
        token_lens = np.array([len(token_data) for token_data in tokenized_data])

        if len(token_lens) == 0:
            return []

        # Calculate start indices for masking
        # astype(int) performs floor operation by truncation
        num_future_to_mask = (rate * token_lens).astype(int)
        start_mask_indices = np.maximum(1, token_lens - num_future_to_mask)

        # Handle edge cases
        mask_valid = token_lens > 1  # Only cells with >1 timestep can be masked
        start_mask_indices = np.where(mask_valid, start_mask_indices, token_lens)

        # Create masks with list comprehension
        # Needed to handle variable lengths
        full_mask = [
            (
                np.concatenate(
                    [
                        np.zeros(start_idx, dtype=bool),
                        np.ones(max(0, token_len - start_idx), dtype=bool),
                    ]
                )
                if token_len > 1
                else (np.zeros(1, dtype=bool) if token_len == 1 else np.array([], dtype=bool))
            )
            for token_len, start_idx in zip(token_lens, start_mask_indices, strict=False)
        ]

        return full_mask

    def build_samples_for_stream(
        self, training_mode: str, num_cells: int, training_cfg: dict
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Construct teacher/student keep masks for a stream.
        SampleMetaData is currently just a dict with the masking params used.
        """

        target_cfgs = training_cfg.get("target_input", [])
        source_cfgs = training_cfg.get("model_input", [])

        # target and source are assumed identical when target is not specified
        if len(target_cfgs) == 0:
            target_cfgs = source_cfgs

        # iterate over all target samples
        target_masks: list[np.typing.NDArray] = []
        target_metadata: list[SampleMetaData] = []
        target_config_mapping = []  # Track which config each target mask came from
        # different strategies
        for i_target, target_cfg in enumerate(target_cfgs):
            # different samples/view per strategy
            for _ in range(target_cfg.get("num_samples", 1)):
                target_mask, mask_params = self._get_mask(
                    num_cells=num_cells,
                    strategy=target_cfg.get("masking_strategy"),
                    target_mask=None,
                    masking_strategy_config=target_cfg.get("masking_strategy_config", {}),
                )
                target_masks += [target_mask]
                target_metadata += [SampleMetaData(params={**target_cfg, **mask_params})]
                target_config_mapping += [i_target]  # Track which config this mask came from

        # iterate over all source samples
        source_masks: list[np.typing.NDArray] = []
        source_metadata: list[SampleMetaData] = []
        source_target_mapping = []
        source_config_mapping = []  # Track which config each source mask came from
        # different strategies
        for i_source, source_cfg in enumerate(source_cfgs):
            # samples per strategy
            for _ in range(source_cfg.get("num_samples", 1)):
                # Prepare masking config - inject target mask if overlap_ratio is specified
                masking_cfg = source_cfg.get("masking_strategy_config", {}).copy()
                if "overlap_ratio" in masking_cfg and len(target_masks) > 0:
                    # Enable overlap control by passing teacher's mask
                    target_mask_for_overlap = target_masks[i_source % len(target_masks)]
                    masking_cfg["overlap_with_mask"] = target_mask_for_overlap.cpu().numpy() if hasattr(target_mask_for_overlap, 'cpu') else target_mask_for_overlap

                source_mask, mask_params = self._get_mask(
                    num_cells=num_cells,
                    strategy=source_cfg.get("masking_strategy"),
                    masking_strategy_config=masking_cfg,
                    target_mask=target_masks[i_source % len(target_masks)],
                    relationship=source_cfg.get("relationship", "independent"),
                )
                source_masks += [source_mask]
                source_metadata += [SampleMetaData(params={**target_cfg, **mask_params})]
                # TODO: proper correspondence between source and target
                source_target_mapping += [i_source % len(target_masks)]
                source_config_mapping += [i_source]  # Track which config this mask came from

        source_target_mapping = np.array(source_target_mapping, dtype=np.int32)
        source_config_mapping = np.array(source_config_mapping, dtype=np.int32)
        target_config_mapping = np.array(target_config_mapping, dtype=np.int32)

        return (
            (target_masks, target_metadata, target_config_mapping),
            (source_masks, source_metadata, source_config_mapping),
            source_target_mapping,
        )

    def _get_mask(
        self,
        num_cells: int,
        strategy: str | None = None,
        masking_strategy_config: dict | None = None,
        target_mask: np.typing.NDArray | None = None,
        relationship: str = "subset",
    ) -> (np.typing.NDArray, dict):
        """Get effective mask, combining with target mask if specified.

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: currently supports 'random' and 'healpix'. Uses
            instance default if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).
        constraint_keep_mask : np.ndarray | None
            Optional boolean mask of allowed cells (True = allowed). Selection will be
            limited to these cells. For subset/disjoint relationships.

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        dict
            Parameters describing the masking that was applied
        """

        # handle cases where mask is directly derived from target_mask
        if target_mask is not None:
            if relationship == "complement":
                mask = ~target_mask
                return mask, {}

        # get mask
        mask, params = self._generate_cell_mask(num_cells, strategy, masking_strategy_config)

        # handle cases where mask needs to be combined with target_mask
        if target_mask is not None:
            if relationship == "subset":
                mask = mask & target_mask
            elif relationship == "disjoint":
                mask = mask & (~target_mask)

        return (mask, params)

    def _generate_cell_mask(
        self, num_cells: int, strategy: str, masking_strategy_config: dict
    ) -> (np.typing.NDArray, dict):
        """Generate a boolean keep mask at data healpix level (True = keep cell).

        Parameters
        ----------
        num_cells : int
            Number of cells at data level (should equal 12 * 4**healpix_level).
        strategy : str | None
            Cell selection strategy: currently supports 'random' and 'healpix'. Uses
            instance default if None.
        masking_strategy_config : dict | None
            Optional override of strategy config (e.g., {'hl_mask': 3}).
        constraint_keep_mask : np.ndarray | None
            Optional boolean mask of allowed cells (True = allowed). Selection will be
            limited to these cells. For subset/disjoint relationships.

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        """

        # params describing the masking
        masking_params = {}

        # get config for mask

        cfg = masking_strategy_config
        keep_rate = cfg.get("rate", None)
        assert keep_rate is not None, 'No sampling rate "rate" specified.'

        # sample rate if requested (only if explicit rate not provided)
        # if rate is None and self.masking_rate_sampling:
        #     keep_rate = self._get_sampling_rate()

        assert 0.0 <= keep_rate <= 1.0, f"keep_rate out of bounds: {keep_rate}"
        assert num_cells == self.healpix_num_cells, (
            "num_cells inconsistent with configured healpix level."
        )

        # generate cell mask

        if strategy == "random":
            mask = self.rng.uniform(0, 1, num_cells) < keep_rate

        elif "forecast" in strategy or strategy == "causal":
            mask = np.ones(num_cells, dtype=np.bool)

            if "diffusion_rn" in masking_strategy_config:
                masking_params["noise_level_rn"] = self.rng.normal(0.0, 1.0)

        elif strategy == "healpix":
            hl_data = self.healpix_level_data
            hl_mask = cfg.get("hl_mask")
            assert hl_mask is not None and hl_mask < hl_data, (
                "For healpix keep mask generation, cfg['hl_mask'] must be set and < data level."
            )
            num_parent_cells = 12 * (4**hl_mask)
            level_diff = hl_data - hl_mask
            num_children_per_parent = 4**level_diff
            # number of parents to KEEP
            num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))
            if num_parents_to_keep == 0:
                mask = np.zeros(num_cells, dtype=bool)
            else:
                parent_ids = self.rng.choice(num_parent_cells, num_parents_to_keep, replace=False)
                child_offsets = np.arange(num_children_per_parent)
                child_indices = (
                    parent_ids[:, None] * num_children_per_parent + child_offsets
                ).reshape(-1)
                mask = np.zeros(num_cells, dtype=bool)
                mask[child_indices] = True

        elif strategy == "cropping_healpix":
            # Spatial cropping: select spatially contiguous region and KEEP it (mask everything else)
            # This is the elegant inverse of healpix masking
            hl_data = self.healpix_level_data
            hl_mask = cfg.get("hl_mask")
            assert hl_mask is not None and hl_mask < hl_data, (
                "For cropping_healpix, cfg['hl_mask'] must be set and < data level."
            )
            num_parent_cells = 12 * (4**hl_mask)
            level_diff = hl_data - hl_mask
            num_children_per_parent = 4**level_diff

            # Number of parents to keep (spatially contiguous)
            num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))

            if num_parents_to_keep == 0:
                mask = np.zeros(num_cells, dtype=bool)
            else:
                # Spatial selection method
                method = cfg.get("method", "geodesic_disk")  # Default to best method for SSL

                # Deterministic overlap support (for IBOT-style training)
                # overlap_with_mask: Teacher's crop mask at data level (hl=5)
                # overlap_ratio: Target overlap as fraction of student crop [0.0-1.0]
                overlap_with_mask = cfg.get("overlap_with_mask", None)
                overlap_ratio = cfg.get("overlap_ratio", None)

                # If overlap is requested but no mask provided, log a warning
                if overlap_ratio is not None and overlap_with_mask is None:
                    _logger.warning(
                        "overlap_ratio specified but no overlap_with_mask provided. "
                        "Overlap control requires reference mask from previous crop. "
                        "Use programmatic API for controlled overlap."
                    )

                # NEW DETERMINISTIC APPROACH: Work at data level (hl=5) for exact overlap
                if overlap_with_mask is not None and overlap_ratio is not None:
                    assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio must be in [0.0, 1.0]"

                    # Calculate total cells we want in student crop
                    total_student_cells = num_parents_to_keep * num_children_per_parent

                    # Get indices of teacher's cells (at data level)
                    teacher_cell_indices = np.where(overlap_with_mask)[0]
                    non_teacher_cell_indices = np.where(~overlap_with_mask)[0]

                    # Deterministically select overlap cells from teacher
                    num_overlap_cells = int(np.round(overlap_ratio * total_student_cells))
                    num_overlap_cells = min(num_overlap_cells, len(teacher_cell_indices))  # Can't exceed teacher size

                    # Randomly select which teacher cells to use for overlap
                    if num_overlap_cells > 0:
                        overlap_cell_indices = self.rng.choice(
                            teacher_cell_indices,
                            size=num_overlap_cells,
                            replace=False
                        )
                    else:
                        overlap_cell_indices = np.array([], dtype=int)

                    # Select remaining cells from outside teacher crop
                    # Simplified: directly sample from non-teacher cells to guarantee correct count
                    num_non_overlap_cells = total_student_cells - num_overlap_cells

                    if num_non_overlap_cells > 0 and len(non_teacher_cell_indices) > 0:
                        # Directly sample from non-teacher cells at data level
                        # This ensures we get exactly the right number
                        num_available = min(num_non_overlap_cells, len(non_teacher_cell_indices))
                        non_overlap_child_indices = self.rng.choice(
                            non_teacher_cell_indices,
                            size=num_available,
                            replace=False
                        )
                    else:
                        non_overlap_child_indices = np.array([], dtype=int)

                    # Combine overlap and non-overlap cells
                    child_indices = np.concatenate([overlap_cell_indices, non_overlap_child_indices])

                    # Create mask
                    mask = np.zeros(num_cells, dtype=bool)
                    mask[child_indices] = True

                    _logger.info(
                        f"Deterministic overlap: target={overlap_ratio:.1%}, "
                        f"actual={len(overlap_cell_indices)/len(child_indices):.1%} "
                        f"({len(overlap_cell_indices)}/{len(child_indices)} cells)"
                    )

                else:
                    # No overlap control - use standard spatial selection
                    parent_ids = self._select_spatially_contiguous_cells(
                        healpix_level=hl_mask,
                        num_cells_to_select=num_parents_to_keep,
                        center_cell=None,
                        method=method,
                        overlap_with=None,
                        overlap_ratio=None,
                    )

                    # Project to data level
                    child_offsets = np.arange(num_children_per_parent)
                    child_indices = (
                        parent_ids[:, None] * num_children_per_parent + child_offsets
                    ).reshape(-1)

                    # Create mask: True = KEEP (the crop), False = MASK (everything else)
                    mask = np.zeros(num_cells, dtype=bool)
                    mask[child_indices] = True

        else:
            raise NotImplementedError(
                f"Cell selection strategy '{strategy}' not supported for keep mask generation."
            )

        mask = to_bool_tensor(mask)

        return (mask, masking_params)
