import copy
import logging
import warnings

import astropy_healpix as hp
import numpy as np
import omegaconf
import torch
from astropy import units as u
from numpy.typing import NDArray

from weathergen.datasets.batch import SampleMetaData
from weathergen.train.utils import Stage
from weathergen.utils.utils import is_stream_diagnostic, is_stream_forcing

logger = logging.getLogger(__name__)


class MaskData:
    masks: list[np.typing.NDArray] = []
    metadata: list[SampleMetaData] = []

    def __init__(self):
        self.masks = []
        self.metadata = []

    def __len__(self):
        return len(self.masks)

    def add_mask(self, mask, params, cfg, losses, idx, correspondence, relationship):
        self.masks += [mask]
        self.metadata += [
            SampleMetaData(
                params={**cfg, **params},
                mask=mask,
                global_params={
                    "idx": idx,
                    "correspondence": correspondence,
                    "loss": losses,
                    "relationship": relationship,
                },
            )
        ]

    def get_mask(self, idx: int) -> np.typing.NDArray:
        return self.masks[idx]

    def get_params(self, idx: int) -> dict:
        """Get the masking parameters for a specific mask."""
        return self.metadata[idx].params


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


def validate_correspondence_mode(correspondence_mode, target_cfgs, source_cfgs):
    """
    Validate that the configs are consistent with the correspondence mode
    """

    num_target_samples = np.array([t.get("num_samples", 1) for t in target_cfgs]).sum()
    num_source_samples = np.array([s.get("num_samples", 1) for s in source_cfgs]).sum()

    if correspondence_mode == "one-to-one":
        assert len(target_cfgs) == len(source_cfgs), (
            "With target_correspondence_mode mode one-to-one, number of source and target "
            + "strategies has to match."
        )
        assert num_target_samples.item() == num_source_samples.item(), (
            "With target_correspondence_mode mode one-to-one, number of source and target "
            + "samples has to match."
        )

    if correspondence_mode == "equal-split-all":
        assert num_source_samples.item() % num_target_samples.item() == 0, (
            "With target_correspondence_mode mode equal-split-all, number of source samples "
            + "has to be divisible by number of target samples."
        )


# Convert to torch.bool
def to_bool_tensor(arr):
    return torch.from_numpy(np.asarray(arr)).to(torch.bool)


class Masker:
    """Class to generate masks for token sequences and apply them.
    This class supports different masking strategies and combinations.

    Attributes:
        masking_rate (float): The base rate at which tokens are masked.
        masking_strategy (str): The strategy used for masking (e.g., "random",
        "healpix", "cropping_healpix").
        current_strategy (str): The current strategy in use, relevant
                                when using "combination" strategy.
        "random" - random masking of tokens at the level of the data
        "healpix" - masking at the level of HEALPix cells, where all child cells
                    of a parent cell at a specific HEALpix level are masked
                    if the parent is masked.
                    The healpix level must be configured with hl_mask.
                    e.g. masking_strategy_config = {"hl_mask": 1}
                    with hl_mask the level for masking that we want to apply
                    e.g. level 1 very large cells masked
        "cropping_healpix" - spatial cropping that keeps spatially contiguous regions
                    and masks everything else. Uses neighbor relationships or geodesic
                    distance to ensure spatial contiguity. For DINO/JEPA/IBOT.
                    e.g. masking_strategy_config = {"hl_mask": 0, "method": "geodesic_disk"}
                    method: "disk" (neighbor growth), "random_walk", or "geodesic_disk" (circular)
        masking_rate_sampling (bool): Whether to sample the masking rate from a distribution.
        masking_strategy_config (dict): Configuration for the masking strategy, can include
                                        additional parameters like "hl_mask", etc.
                                        specific to the masking strategy. See above.
    """

    def __init__(self, healpix_level: int, stage: Stage):
        self.rng = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_level_data = healpix_level
        self.healpix_num_cells = 12 * (4**healpix_level)
        self._hp_cache = {}

        self.stage = stage

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.rng = rng

    def _get_sampling_rate(self, cfg):
        """
        Get the sampling, if requested by sampling it itself
        """

        rate = cfg.get("rate", None)
        assert rate is not None, 'No sampling rate "rate" specified.'

        if cfg.get("rate_sampling", False):
            rate = np.clip(
                np.abs(self.rng.normal(loc=rate, scale=1.0 / (2.5 * np.pi))),
                0.01,
                0.99,
            )
        assert 0.0 <= rate <= 1.0, f"keep_rate out of bounds: {rate}"

        return rate

    def get_target_rel_mask(self, target_masks, masking_config):
        """
        Get target relationship strategy and target mask
        """
        relationship = masking_config.get("target_relationship", {"independent": None})
        assert len(relationship) == 1, "Only one target_relationship supported."

        target_idx = list(relationship.values())[0]

        target_relationship_mask = (
            list(relationship.keys())[0],  # target relationship strategy
            target_masks.get_mask(target_idx),  # target mask
        )

        return target_relationship_mask, target_idx

    def parse_src_target_correspondence(self, losses, target_cfgs, source_cfgs) -> dict:
        """
        Parses losses and obtain consolidated source -> target correspondence dict
        """

        # collect target-source correspondence for all loss terms
        corrs = []
        for _, loss_term in losses.items():
            for loss_name, loss_fct in loss_term.loss_fcts.items():
                corr = loss_fct.get("target_source_correspondence", None)

                # correspondence not specified; falling back to default 1-to-1 correspondence
                # at the level of the configs
                if corr is None:
                    assert len(target_cfgs) == len(source_cfgs), (
                        "No source/target correspondence specified but number of source and target "
                        + "configs also not matching."
                    )
                    corr = dict([(i, i) for i in range(len(target_cfgs))])

                corr_dict = {}
                for target_idx, source_spec in corr.items():
                    # process into common long format
                    target_idx = int(target_idx)
                    if type(source_spec) is omegaconf.dictconfig.DictConfig:
                        # TODO: check format of dict
                        # append loss_name
                        corr_dict[target_idx] = dict(
                            [(int(k), (v, loss_name)) for k, v in source_spec.items()]
                        )
                    elif type(source_spec) is omegaconf.listconfig.ListConfig:
                        corr_dict[target_idx] = dict(
                            [(int(v), (None, loss_name)) for v in source_spec]
                        )
                    elif type(source_spec) is int:
                        corr_dict[target_idx] = {source_spec: (None, loss_name)}
                    else:
                        assert False, (
                            "Invalid target_source_correspondence specification. Needs to be "
                            + "integer corresponding to a specific source, list of source or a "
                            + "dictionary specifying the correspondence."
                        )

                corrs += [corr_dict]

        # check that all target/sources indices are ints; conf can have type mismatches due to
        # conf merging
        are_ints = np.array(
            [
                [type(k) is int and type(next(iter(v.keys()))) is int for k, v in corr.items()]
                for corr in corrs
            ]
        ).all()
        assert are_ints, "error parsing correspondence, all indices must be int"

        # merge correspondences
        corr_dict = {}
        for k_target in range(len(target_cfgs)):
            # require identical relationship type when target has same source correspondence in
            # different loss terms
            vs = [c.get(k_target) for c in corrs if c.get(k_target) is not None]
            vs_ks_unique = list(set([kk for v in vs for kk in list(v.keys())]))
            for k_source in vs_ks_unique:
                rel_loss = [v.get(k_source) for v in vs if v.get(k_source) is not None]
                # check that specified relationship is consistent
                assert len(list(set([rl[0] for rl in rel_loss]))) == 1, (
                    "Inconsistent target_source correspondence: one source has multiple target "
                    + "with different source/target relationships"
                )
                if k_source >= len(source_cfgs):
                    logger.warning(
                        f"target_source_correspondence contains non-existent source {k_source}."
                    )
                    continue
                if k_target >= len(target_cfgs):
                    logger.warning(
                        f"target_source_correspondence contains non-existent source {k_target}."
                    )
                    continue
                # add valid entry, source-target pair can have multiple losses
                losses = [rl[1] for rl in rel_loss]
                # add, making sure that each source has only one target (subset relationships
                # but also physical loss )
                assert corr_dict.get(k_source) is None, "source cfg needs unique target"
                corr_dict[k_source] = (k_target, (rel_loss[0][0], losses))

        # TODO: check validity of target_source_correspondence with target and source cfgs

        return corr_dict

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        stage_cfg: dict,
        stream_cfg: dict,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Construct teacher/student keep masks for a stream.
        SampleMetaData is currently just a dict with the masking params used.
        """

        # target and source configs
        target_cfgs = stage_cfg.get("target_input", [])
        source_cfgs = stage_cfg.get("model_input", [])

        # target and source are assumed identical when target is not specified
        if len(target_cfgs) == 0:
            target_cfgs = copy.deepcopy(source_cfgs)

        losses = stage_cfg.losses
        corr_dict = self.parse_src_target_correspondence(losses, target_cfgs, source_cfgs)

        target_masks = MaskData()

        # iterate over all target samples
        # different strategies
        i_target = 0
        for i_cfg, (_, target_cfg) in enumerate(target_cfgs.items()):
            # different samples/view per strategy
            for _ in range(target_cfg.get("num_samples", 1)):
                # determine if forcing dataset => mask is empty
                if is_stream_forcing(stream_cfg, self.stage):
                    target_mask, mask_params = torch.zeros(num_cells, dtype=torch.bool), {}
                else:
                    target_mask, mask_params = self._get_mask(
                        num_cells=num_cells,
                        strategy=target_cfg.get("masking_strategy"),
                        masking_strategy_config=target_cfg.get("masking_strategy_config", {}),
                        target_relationship_mask=("independent", None),
                    )

                # get all losses and flatten
                losses = [v[1][1] for _, v in corr_dict.items() if len(v) > 0 and v[0] == i_cfg]
                losses = [ll for lt in losses for ll in lt]
                # corresponding sources
                corr = [k for k, v in corr_dict.items() if len(v) > 0 and v[0] == i_cfg]
                # skip items that do not appear in loss
                if len(corr) == 0:
                    continue
                # add
                target_masks.add_mask(
                    target_mask, mask_params, target_cfg, losses, i_target, corr, None
                )
                i_target += 1

        source_masks = MaskData()
        source_target_mapping = []
        target_num_samples = get_num_samples(target_cfgs)
        i_source = 0
        for i_src_cfg, (_, source_cfg) in enumerate(source_cfgs.items()):
            # skip items that do not appear in loss
            if i_src_cfg not in corr_dict:
                continue
            # samples per strategy
            for i_sample in range(source_cfg.get("num_samples", 1)):
                masking_config = source_cfg.get("masking_strategy_config", {})
                # extract corresponding target
                target_cfg_idx, rel_losses = corr_dict[i_src_cfg]
                relationship, losses = rel_losses
                # ensure proper default relationships
                if relationship is None:
                    if source_cfg.get("masking_strategy") == "random":
                        # default for masked token modeling
                        relationship = "complement"
                    else:
                        # default for forecasting
                        relationship = "independent"
                target_idx = target_num_samples[:target_cfg_idx].sum()
                # iterate sequentially through targets (to enable 1-to-1 correspondence when no
                # target is specified)
                target_idx += i_sample % target_num_samples[target_cfg_idx].item()

                # Get target metadata for relationships that need geometric info
                target_metadata = target_masks.get_params(target_idx)

                # determine if forcing dataset => mask is empty
                if is_stream_diagnostic(stream_cfg, self.stage):
                    source_mask, mask_params = torch.zeros(num_cells, dtype=torch.bool), {}
                else:
                    # Get target metadata for relationships that need geometric info
                    target_metadata = target_masks.get_params(target_idx)

                    source_mask, mask_params = self._get_mask(
                        num_cells=num_cells,
                        strategy=source_cfg.get("masking_strategy"),
                        masking_strategy_config=masking_config,
                        target_relationship_mask=(relationship, target_masks.get_mask(target_idx)),
                        target_metadata=target_metadata,
                    )
                corr = target_idx
                source_masks.add_mask(
                    source_mask, mask_params, source_cfg, losses, i_source, corr, relationship
                )

                source_target_mapping += [target_idx]
                i_source += 1

        source_target_mapping = np.array(source_target_mapping, dtype=np.int32)

        return (target_masks, source_masks, source_target_mapping)

    def _get_mask(
        self,
        num_cells: int,
        strategy: str,
        masking_strategy_config: dict,
        target_relationship_mask: (str, np.typing.NDArray),
        target_metadata: dict | None = None,
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

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        dict
            Parameters describing the masking that was applied
        """

        relationship, target_mask = target_relationship_mask

        if strategy == "forecast":
            if relationship is not None:
                assert relationship == "independent", (
                    "strategy forecast requires relationship independent "
                )

        # handle cases where mask is directly derived from target_mask
        if relationship == "complement":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = ~target_mask
            return mask, {}
        elif relationship == "identity":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = target_mask
            params = {}

        elif relationship == "cone_distance":
            mask, params = self._get_cone_distance_relationship_mask(
                num_cells,
                masking_strategy_config,
                target_mask,
                target_metadata,
            )
        elif relationship == "contained_cone":
            mask, params = self._get_contained_cone_relationship_mask(
                num_cells,
                masking_strategy_config,
                target_mask,
                target_metadata,
            )
        elif relationship == "separated_cone":
            mask, params = self._get_separated_cone_relationship_mask(
                num_cells,
                masking_strategy_config,
                target_mask,
                target_metadata,
            )
        else:
            # get mask
            mask, params = self._generate_cell_mask(num_cells, strategy, masking_strategy_config)

            mask, params = self._apply_subset_or_disjoint_relationship(
                num_cells,
                strategy,
                relationship,
                masking_strategy_config,
                target_mask,
                target_metadata,
                mask,
                params,
            )

        return mask, params

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

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        """

        # params describing the masking
        masking_params = {}

        assert num_cells == self.healpix_num_cells, (
            "num_cells inconsistent with configured healpix level."
        )

        # generate cell mask

        if strategy == "random":
            keep_rate = self._get_sampling_rate(masking_strategy_config)
            mask = self.rng.uniform(0, 1, num_cells) < keep_rate

        elif "forecast" in strategy or strategy == "causal":
            mask = np.ones(num_cells, dtype=np.bool)

            if "diffusion_rn" in masking_strategy_config:
                masking_params["noise_level_rn"] = self.rng.normal(0.0, 1.0)

        elif strategy == "healpix":
            # prepare healpix-based masking
            keep_rate = self._get_sampling_rate(masking_strategy_config)
            hl_mask, num_parent_cells, num_children_per_parent, num_parents_to_keep = (
                self._prepare_healpix_based_masking(masking_strategy_config, keep_rate)
            )

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

        # Spatial healpix based cropping, select contiguous region
        elif strategy == "cropping_healpix":
            # prepare healpix-based masking
            keep_rate = self._get_sampling_rate(masking_strategy_config)
            hl_mask, num_parent_cells, num_children_per_parent, num_parents_to_keep = (
                self._prepare_healpix_based_masking(masking_strategy_config, keep_rate)
            )

            if num_parents_to_keep == 0:
                mask = np.zeros(num_cells, dtype=bool)
            else:
                # Spatial selection method
                method = masking_strategy_config.get("method", "geodesic_disk")

                # Compute anchored center cell if configured (for regional dataset focus)
                anchor_center_cell = None
                anchor_lat = masking_strategy_config.get("anchor_latitude")
                anchor_lon = masking_strategy_config.get("anchor_longitude")

                if anchor_lat is not None and anchor_lon is not None:
                    anchor_lat_rad = np.radians(anchor_lat)
                    anchor_lon_rad = np.radians(anchor_lon)

                    # Apply jitter around anchor if configured
                    jitter_deg = masking_strategy_config.get("anchor_jitter_degrees", 0.0)
                    if jitter_deg > 0:
                        jitter_distance = self.rng.uniform(0, np.radians(jitter_deg))
                        jitter_azimuth = self.rng.uniform(0, 2 * np.pi)
                        anchor_lon_rad, anchor_lat_rad = self._get_destination_latlon(
                            anchor_lon_rad, anchor_lat_rad, jitter_distance, jitter_azimuth
                        )

                    hp_obj = self._get_hp_obj(hl_mask)
                    anchor_center_cell = int(
                        hp_obj.lonlat_to_healpix(anchor_lon_rad * u.rad, anchor_lat_rad * u.rad)
                    )

                # Use standard spatial selection - returns (mask, center_cell) tuple
                mask, center_cell = self._select_spatially_contiguous_cells(
                    healpix_level=hl_mask,
                    num_cells=num_cells,
                    num_cells_to_select=num_parents_to_keep,
                    num_children_per_parent=num_children_per_parent,
                    center_cell=anchor_center_cell,
                    method=method,
                )

                # Store geometry info for downstream relationships
                # (e.g., cone_distance, contained_cone, etc.)
                masking_params["center_cell"] = center_cell
                masking_params["hl_mask"] = hl_mask
                masking_params["rate"] = keep_rate

        else:
            raise NotImplementedError(
                f"Cell selection strategy '{strategy}' not supported for keep mask generation."
            )

        mask = to_bool_tensor(mask)

        return (mask, masking_params)

    def _get_cone_relationship_params(
        self, masking_strategy_config: dict, center_cell: int
    ) -> dict:
        """Get standard metadata returned by cone-based relationships."""
        return {
            "center_cell": center_cell,
            "hl_mask": masking_strategy_config.get("hl_mask", 0),
            "rate": masking_strategy_config.get("rate", 0.5),
        }

    def _get_teacher_cone_metadata(
        self,
        relationship: str,
        masking_strategy_config: dict,
        target_metadata: dict,
        require_rate: bool = False,
    ) -> tuple[int, float | None, int]:
        """Get teacher cone geometry from explicit metadata."""
        teacher_center_cell = target_metadata.get("center_cell")
        teacher_rate = target_metadata.get("rate")
        teacher_hl_mask = target_metadata.get("hl_mask", masking_strategy_config.get("hl_mask", 0))

        assert teacher_center_cell is not None, (
            f"relationship '{relationship}' requires 'center_cell' in target_metadata"
        )
        if require_rate:
            assert teacher_rate is not None, (
                f"relationship '{relationship}' requires 'rate' in target_metadata"
            )

        return teacher_center_cell, teacher_rate, teacher_hl_mask

    def _get_center_distance_degrees(self, masking_strategy_config: dict) -> float:
        """Get cone distance from config."""
        # Get cone distance parameter - supports fixed value or random selection
        center_distance_degrees_random = masking_strategy_config.get(
            "center_distance_degrees_random", False
        )
        if center_distance_degrees_random:
            # Random selection from range with specified step (default: 0 to 90 in steps of 15)
            min_dist = masking_strategy_config.get("center_distance_degrees_min", 0)
            max_dist = masking_strategy_config.get("center_distance_degrees_max", 90)
            step = masking_strategy_config.get("center_distance_degrees_step", 15)
            possible_values = list(range(min_dist, max_dist + 1, step))
            return float(self.rng.choice(possible_values))

        center_distance_degrees = masking_strategy_config.get("center_distance_degrees")
        assert center_distance_degrees is not None, (
            "relationship 'cone_distance' requires 'center_distance_degrees' or "
            "'center_distance_degrees_random: true' in config"
        )

        return center_distance_degrees

    def _get_cone_distance_relationship_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        target_mask: np.typing.NDArray | None,
        target_metadata: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        """Handle the cone_distance relationship."""
        assert target_mask is not None, "relationship 'cone_distance' requires target_mask"
        assert target_metadata is not None, "relationship 'cone_distance' requires target_metadata"

        # Get cone distance parameter
        center_distance_degrees = self._get_center_distance_degrees(masking_strategy_config)

        # Get teacher center cell from explicit metadata (not instance state)
        teacher_center_cell, _, teacher_hl_mask = self._get_teacher_cone_metadata(
            "cone_distance",
            masking_strategy_config,
            target_metadata,
        )

        # Create cone at specified distance from teacher
        mask, student_center_cell = self._create_cone_distance_mask(
            num_cells,
            masking_strategy_config,
            teacher_center_cell,
            center_distance_degrees,
            teacher_hl_mask,
        )
        params = self._get_cone_relationship_params(
            masking_strategy_config,
            student_center_cell,
        )
        return mask, params

    def _get_contained_cone_relationship_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        target_mask: np.typing.NDArray | None,
        target_metadata: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        """Handle the contained_cone relationship."""
        assert target_mask is not None, "relationship 'contained_cone' requires target_mask"
        assert target_metadata is not None, "relationship 'contained_cone' requires target_metadata"

        # Get teacher geometry
        teacher_center_cell, teacher_rate, teacher_hl_mask = self._get_teacher_cone_metadata(
            "contained_cone",
            masking_strategy_config,
            target_metadata,
            require_rate=True,
        )

        mask, student_center_cell = self._create_contained_cone_mask(
            num_cells,
            masking_strategy_config,
            teacher_center_cell,
            teacher_rate,
            teacher_hl_mask,
        )
        params = self._get_cone_relationship_params(
            masking_strategy_config,
            student_center_cell,
        )
        return mask, params

    def _get_separated_cone_relationship_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        target_mask: np.typing.NDArray | None,
        target_metadata: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        """Handle the separated_cone relationship."""
        assert target_mask is not None, "relationship 'separated_cone' requires target_mask"
        assert target_metadata is not None, "relationship 'separated_cone' requires target_metadata"

        # Get teacher geometry
        teacher_center_cell, teacher_rate, teacher_hl_mask = self._get_teacher_cone_metadata(
            "separated_cone",
            masking_strategy_config,
            target_metadata,
            require_rate=True,
        )

        mask, student_center_cell = self._create_separated_cone_mask(
            num_cells,
            masking_strategy_config,
            teacher_center_cell,
            teacher_rate,
            teacher_hl_mask,
        )
        params = self._get_cone_relationship_params(
            masking_strategy_config,
            student_center_cell,
        )
        return mask, params

    def _has_geometry_aware_target_metadata(
        self, strategy: str, target_metadata: dict | None
    ) -> bool:
        """Check whether geometry-aware subset/disjoint handling is available."""
        return (
            strategy == "cropping_healpix"
            and target_metadata is not None
            and target_metadata.get("center_cell") is not None
            and target_metadata.get("rate") is not None
        )

    def _apply_subset_or_disjoint_relationship(
        self,
        num_cells: int,
        strategy: str,
        relationship: str,
        masking_strategy_config: dict,
        target_mask: np.typing.NDArray | None,
        target_metadata: dict | None,
        mask: np.typing.NDArray,
        params: dict,
    ) -> tuple[np.typing.NDArray, dict]:
        """Apply subset/disjoint logic after mask generation."""
        if relationship not in {"subset", "disjoint"}:
            return mask, params

        if relationship == "subset":
            assert target_mask is not None, (
                f"relationship: {relationship} incompatible with target_mask None"
            )

            # Check if geometry-aware processing is available
            if self._has_geometry_aware_target_metadata(strategy, target_metadata):
                # Use geometry-aware contained_cone for spatial contiguity
                logger.debug("Using geometry-aware contained_cone for subset with cropping_healpix")
                teacher_center_cell, teacher_rate, teacher_hl_mask = (
                    self._get_teacher_cone_metadata(
                        relationship,
                        masking_strategy_config,
                        target_metadata,
                        require_rate=True,
                    )
                )

                mask, center_cell = self._create_contained_cone_mask(
                    num_cells,
                    masking_strategy_config,
                    teacher_center_cell,
                    teacher_rate,
                    teacher_hl_mask,
                )
                params.update(
                    self._get_cone_relationship_params(masking_strategy_config, center_cell)
                )
            else:
                # Standard boolean AND for non-cropping strategies
                mask = mask & target_mask
        else:
            assert target_mask is not None, (
                f"relationship: {relationship} incompatible with target_mask None"
            )

            # Check if geometry-aware processing is available
            if self._has_geometry_aware_target_metadata(strategy, target_metadata):
                # Use geometry-aware separated_cone for spatial contiguity
                logger.debug(
                    "Using geometry-aware separated_cone for disjoint with cropping_healpix"
                )
                teacher_center_cell, teacher_rate, teacher_hl_mask = (
                    self._get_teacher_cone_metadata(
                        relationship,
                        masking_strategy_config,
                        target_metadata,
                        require_rate=True,
                    )
                )

                mask, center_cell = self._create_separated_cone_mask(
                    num_cells,
                    masking_strategy_config,
                    teacher_center_cell,
                    teacher_rate,
                    teacher_hl_mask,
                )
                params.update(
                    self._get_cone_relationship_params(masking_strategy_config, center_cell)
                )
            else:
                # Standard boolean AND NOT for non-cropping strategies
                mask = mask & (~target_mask)

        return mask, params

    def _select_spatially_contiguous_cells(
        self,
        healpix_level: int,
        num_cells: int,
        num_cells_to_select: int,
        num_children_per_parent: int,
        center_cell: int | None = None,
        method: str = "geodesic_disk",
    ) -> tuple[NDArray, int]:
        """
        Select spatially contiguous cells on the sphere using neighbor relationships.

        This is the core spatial selection helper used for both masking and cropping.

        Args:
            healpix_level: HEALPix level for selection
            num_cells: Total number of cells at data level
            num_cells_to_select: Number of cells to select
            num_children_per_parent: Number of child cells per parent cell
            center_cell: Starting cell (None = random)
            method: Selection method:
                - "disk": Layer-by-layer neighbor growth (compact regions)
                - "random_walk": Random neighbor selection (irregular shapes)
                - "geodesic_disk": Angular distance selection (circular regions)

        Returns:
            Tuple of (mask, center_cell):
                - mask: Boolean array where True indicates the cell is kept
                - center_cell: The HEALPix cell index used as the cone center
        """

        num_total_cells = 12 * (4**healpix_level)
        nside = 2**healpix_level

        assert num_cells_to_select <= num_total_cells

        # Random starting point if not specified
        if center_cell is None:
            center_cell = self.rng.integers(0, num_total_cells)

        if method == "disk":
            selected = self._select_disk(center_cell, num_cells_to_select, nside)
        elif method == "random_walk":
            selected = self._select_random_walk(center_cell, num_cells_to_select, nside)
        elif method == "geodesic_disk":
            selected = self._select_geodesic_disk(
                center_cell, num_cells_to_select, nside, num_total_cells
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")

        parent_ids = np.array(sorted(selected))

        # Project to data level
        child_offsets = np.arange(num_children_per_parent)
        child_indices = (parent_ids[:, None] * num_children_per_parent + child_offsets).reshape(-1)

        # Create mask: True = MASK (masked tokens), False = KEEP (kept tokens)
        mask = np.zeros(num_cells, dtype=bool)
        mask[child_indices] = True

        return mask, int(center_cell)

    # separate functions for the different methods of producing spatially contiguous regions
    def _select_disk(self, center_cell: int, num_cells_to_select: int, nside: int) -> set[int]:
        """
        Select cells in a disk shape by expanding layer by layer.
        """
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

        return selected

    def _select_random_walk(
        self, center_cell: int, num_cells_to_select: int, nside: int
    ) -> set[int]:
        """
        Random walk through neighbors, creates elongated irregular regions
        """
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

        return selected

    def _select_geodesic_disk(
        self, center_cell: int, num_cells_to_select: int, nside: int, num_total_cells: int
    ) -> set:
        """
        Angular distance selection, creates most uniform somewhat circular regions
        """

        def lonlat_to_xyz(lon, lat):
            """
            Convert lon/lat to 3D cartesian coordinates.
            """
            return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

        # Get center coordinates
        center_lonlat = hp.healpix_to_lonlat(center_cell, nside, order="nested")
        center_lon = float(
            center_lonlat[0].value if hasattr(center_lonlat[0], "value") else center_lonlat[0]
        )
        center_lat = float(
            center_lonlat[1].value if hasattr(center_lonlat[1], "value") else center_lonlat[1]
        )
        center_xyz = lonlat_to_xyz(center_lon, center_lat)

        # Get all cell coordinates
        all_indices = np.arange(num_total_cells)
        all_lonlat = hp.healpix_to_lonlat(all_indices, nside, order="nested")
        all_lon = all_lonlat[0].value if hasattr(all_lonlat[0], "value") else all_lonlat[0]
        all_lat = all_lonlat[1].value if hasattr(all_lonlat[1], "value") else all_lonlat[1]

        all_xyz = np.stack(
            [
                np.cos(all_lat) * np.cos(all_lon),
                np.cos(all_lat) * np.sin(all_lon),
                np.sin(all_lat),
            ],
            axis=1,
        )
        # Compute angular distances and select closest cells
        dot_products = np.clip(np.dot(all_xyz, center_xyz), -1.0, 1.0)
        angular_distances = np.arccos(dot_products)
        selected = np.argsort(angular_distances)[:num_cells_to_select]

        return selected

    def _prepare_healpix_based_masking(self, cfg, keep_rate):
        """
        Prepare healpix masking related attributes.
        """

        hl_data = self.healpix_level_data
        hl_mask = cfg.get("hl_mask")
        assert hl_mask is not None and hl_mask < hl_data, (
            "For healpix keep mask generation, cfg['hl_mask'] must be set and < data level."
        )
        num_parent_cells = 12 * (4**hl_mask)
        level_diff = hl_data - hl_mask
        num_children_per_parent = 4**level_diff
        # number of parents to keep
        num_parents_to_keep = int(np.round(keep_rate * num_parent_cells))

        return hl_mask, num_parent_cells, num_children_per_parent, num_parents_to_keep

    def _get_hp_obj(self, healpix_level: int) -> hp.HEALPix:
        """
        Get cached HEALPix object for efficient repeated queries at the same level.

        Creates and caches HEALPix objects to avoid repeated initialization overhead.

        Args:
            healpix_level: HEALPix resolution level (nside = 2^level)

        Returns:
            Cached HEALPix object configured for NESTED ordering at the specified level

        Note:
            Cache is stored in self._hp_cache dictionary, initialized in __init__
        """
        if healpix_level not in self._hp_cache:
            nside = 2**healpix_level
            self._hp_cache[healpix_level] = hp.HEALPix(nside=nside, order="nested")
        return self._hp_cache[healpix_level]

    def _get_destination_latlon(
        self, origin_lon_rad: float, origin_lat_rad: float, distance_rad: float, azimuth_rad: float
    ) -> tuple[float, float]:
        """
        Compute the destination lon/lat on a sphere from an origin, angular distance,
        and initial azimuth.

        Given a starting point (lon, lat), an angular distance, and an initial bearing,
        return the point reached by traveling along the corresponding great-circle path.
        This is the mathematical foundation for cone distance masking.

        Args:
            origin_lon_rad: Origin longitude in radians [-π, π]
            origin_lat_rad: Origin latitude in radians [-π/2, π/2]
            distance_rad: Angular distance to travel in radians [0, π]
            azimuth_rad: Initial direction of travel in radians [0, 2π]
                        (0 = north, π/2 = east, π = south, 3π/2 = west)

        Returns:
            Tuple of (destination_lon_rad, destination_lat_rad):
                - destination_lon_rad: Destination longitude in radians, normalized to [-π, π]
                - destination_lat_rad: Destination latitude in radians [-π/2, π/2]

        Mathematical Foundation:
            Uses the spherical law of cosines for latitude and the spherical
            law of sines for longitude. See:
            https://www.movable-type.co.uk/scripts/latlong.html
        """
        # Calculate destination latitude using spherical law of cosines
        dest_lat_rad = np.arcsin(
            np.sin(origin_lat_rad) * np.cos(distance_rad)
            + np.cos(origin_lat_rad) * np.sin(distance_rad) * np.cos(azimuth_rad)
        )

        # Calculate destination longitude using spherical law of sines
        dest_lon_rad = origin_lon_rad + np.arctan2(
            np.sin(azimuth_rad) * np.sin(distance_rad) * np.cos(origin_lat_rad),
            np.cos(distance_rad) - np.sin(origin_lat_rad) * np.sin(dest_lat_rad),
        )

        # Normalize longitude to [-π, π] range
        dest_lon_normalized = float(np.arctan2(np.sin(dest_lon_rad), np.cos(dest_lon_rad)))

        return dest_lon_normalized, float(dest_lat_rad)

    def _create_cone_distance_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        teacher_center_cell: int,
        center_distance_degrees: float,
        teacher_hl_mask: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Create student cone at specified angular distance from teacher.

        This creates geometrically controlled overlap where both teacher and student are
        geodesic disks (spatially contiguous circular regions) and their overlap is determined by:
        - The radii of the two cones (from their 'rate' configs)
        - The angular distance between their centers

        Args:
            num_cells: Total cells at data level (12 * 4^healpix_level_data)
            masking_strategy_config: Config for student cone, must contain:
                - 'rate': Fraction of sphere for student cone (e.g., 0.4 = 40%)
                - 'hl_mask': HEALPix level for cone generation
                - 'center_azimuth_degrees' (optional): Direction from teacher (0-360°)
                  If not specified, random direction is chosen
            teacher_center_cell: HEALPix cell index of teacher cone center (teacher_hl_mask level)
            center_distance_degrees: Angular distance between centers (in degrees, 0-180)
            teacher_hl_mask: HEALPix level of the teacher center cell (can differ from student)

        Returns:
            Tuple of (student_mask, student_center_cell):
                - student_mask: Boolean tensor with student cone (geodesic disk)
                - student_center_cell: HEALPix cell index of student cone center (at hl_mask level)

        Mathematical Details:
            Area fraction 'rate' maps to angular radius via spherical cap formula:
            - Spherical cap area = 2πR²(1 - cos(θ)) where R=1 for unit sphere
            - Total sphere area = 4πR² = 4π
            - Area fraction = (1 - cos(θ))/2
            - Solving for θ: radius_rad = arccos(1 - 2*rate)
        """
        # Configuration and Setup
        mask_level = masking_strategy_config.get("hl_mask", 0)
        cone_area_fraction = masking_strategy_config.get("rate", 0.5)
        hp_mask = self._get_hp_obj(mask_level)  # Get cached HEALPix object for STUDENT

        # Calculate Student Cone Center Using Spherical Geometry
        # CRITICAL: Convert teacher center using teacher's hl_mask level, not student's
        # This allows teacher and student to use different hl_mask levels
        hp_teacher = self._get_hp_obj(teacher_hl_mask)  # HEALPix object at TEACHER's level
        teacher_lon, teacher_lat = hp_teacher.healpix_to_lonlat(teacher_center_cell)

        # Determine azimuth (direction from teacher to student)
        azimuth_degrees = masking_strategy_config.get("center_azimuth_degrees")
        if azimuth_degrees is None:
            # Random direction if not specified
            azimuth_degrees = self.rng.uniform(0, 360)
        azimuth_rad = np.deg2rad(azimuth_degrees)
        distance_rad = np.deg2rad(center_distance_degrees)

        # Apply great circle navigation to find student center
        student_lon_rad, student_lat_rad = self._get_destination_latlon(
            teacher_lon.to_value(u.rad), teacher_lat.to_value(u.rad), distance_rad, azimuth_rad
        )

        # Tree-Based Cone Search
        # Convert area fraction to angular radius using spherical cap formula
        # Formula: rate = (1 - cos(θ))/2  =>  θ = arccos(1 - 2*rate)
        cone_radius_rad = np.arccos(1 - 2 * cone_area_fraction)

        # Use tree-based cone search (much faster than computing all distances)
        selected_parent_cells = hp_mask.cone_search_lonlat(
            student_lon_rad * u.rad, student_lat_rad * u.rad, radius=cone_radius_rad * u.rad
        )

        # Get student center cell index
        student_center_cell = int(
            hp_mask.lonlat_to_healpix(student_lon_rad * u.rad, student_lat_rad * u.rad)
        )

        # Efficient Mask Filling Using NESTED Ordering Properties
        # In NESTED ordering, all children of parent P occupy contiguous indices:
        # [P * num_children, (P+1) * num_children)
        # This allows very fast mask filling via slice assignment
        mask_array = np.zeros(num_cells, dtype=bool)
        level_diff = self.healpix_level_data - mask_level

        if level_diff > 0:
            # Multiple data-level cells per mask-level parent
            num_children_per_parent = 4**level_diff
            for parent_idx in selected_parent_cells:
                start_idx = parent_idx * num_children_per_parent
                end_idx = start_idx + num_children_per_parent
                mask_array[start_idx:end_idx] = True
        else:
            # Same level: direct assignment
            mask_array[selected_parent_cells] = True

        return torch.from_numpy(mask_array), student_center_cell

    def _create_contained_cone_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        teacher_center_cell: int,
        teacher_rate: float,
        teacher_hl_mask: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Create student cone that is fully contained within teacher cone.

        This ensures the student geodesic disk is a proper subset of the teacher
        geodesic disk while maintaining spatial contiguity.

        Constraint: student_radius + distance_from_teacher_center <= teacher_radius

        Args:
            num_cells: Total cells at data level
            masking_strategy_config: Config for student cone, must contain:
                - 'rate': Fraction of sphere for student cone
                - 'hl_mask': HEALPix level for cone generation
            teacher_center_cell: HEALPix cell index of teacher cone center
            teacher_rate: Teacher's area fraction (to compute teacher radius)
            teacher_hl_mask: HEALPix level of teacher center cell

        Returns:
            Tuple of (student_mask, student_center_cell)

        Mathematical Details:
            - Teacher radius: theta_T = arccos(1 - 2*teacher_rate)
            - Student radius: theta_S = arccos(1 - 2*student_rate)
            - Maximum distance: d_max = theta_T - theta_S
            - Actual distance: sampled from [0, d_max]
        """
        # Get student parameters
        mask_level = masking_strategy_config.get("hl_mask", 0)
        student_rate = masking_strategy_config.get("rate", 0.5)
        hp_mask = self._get_hp_obj(mask_level)

        # Calculate radii using spherical cap formula
        teacher_radius_rad = np.arccos(1 - 2 * teacher_rate)
        student_radius_rad = np.arccos(1 - 2 * student_rate)

        # Validate constraint is satisfiable
        if student_radius_rad >= teacher_radius_rad:
            raise ValueError(
                f"contained_cone requires student_rate ({student_rate}) < "
                f"teacher_rate ({teacher_rate}). "
                f"Student radius ({np.degrees(student_radius_rad):.1f}deg) >= "
                f"teacher radius ({np.degrees(teacher_radius_rad):.1f}deg)"
            )

        # Maximum distance for containment
        max_distance_rad = teacher_radius_rad - student_radius_rad

        # Sample distance (default: random within valid range)
        distance_mode = masking_strategy_config.get("contained_distance_mode", "random")
        if distance_mode == "random":
            # Uniform in [0, max_distance]
            distance_rad = self.rng.uniform(0, max_distance_rad)
        elif distance_mode == "centered":
            distance_rad = 0.0
        elif distance_mode == "edge":
            distance_rad = max_distance_rad
        else:
            # Fixed fraction of max
            distance_fraction = masking_strategy_config.get("contained_distance_fraction", 0.5)
            distance_rad = distance_fraction * max_distance_rad

        # Get teacher center coordinates
        hp_teacher = self._get_hp_obj(teacher_hl_mask)
        teacher_lon, teacher_lat = hp_teacher.healpix_to_lonlat(teacher_center_cell)

        # Random azimuth direction
        azimuth_rad = self.rng.uniform(0, 2 * np.pi)

        # Compute student center using great circle navigation
        student_lon_rad, student_lat_rad = self._get_destination_latlon(
            teacher_lon.to_value(u.rad),
            teacher_lat.to_value(u.rad),
            distance_rad,
            azimuth_rad,
        )

        # Create student cone using tree-based search
        selected_parent_cells = hp_mask.cone_search_lonlat(
            student_lon_rad * u.rad,
            student_lat_rad * u.rad,
            radius=student_radius_rad * u.rad,
        )

        # Get student center cell index
        student_center_cell = int(
            hp_mask.lonlat_to_healpix(student_lon_rad * u.rad, student_lat_rad * u.rad)
        )

        # Build mask using NESTED ordering properties
        mask_array = np.zeros(num_cells, dtype=bool)
        level_diff = self.healpix_level_data - mask_level

        if level_diff > 0:
            num_children_per_parent = 4**level_diff
            for parent_idx in selected_parent_cells:
                start_idx = parent_idx * num_children_per_parent
                end_idx = start_idx + num_children_per_parent
                mask_array[start_idx:end_idx] = True
        else:
            mask_array[selected_parent_cells] = True

        return torch.from_numpy(mask_array), student_center_cell

    def _create_separated_cone_mask(
        self,
        num_cells: int,
        masking_strategy_config: dict,
        teacher_center_cell: int,
        teacher_rate: float,
        teacher_hl_mask: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Create student cone that is fully separated from teacher cone.

        This ensures the student geodesic disk has no overlap with the teacher
        geodesic disk while maintaining spatial contiguity.

        Constraint: distance_between_centers >= student_radius + teacher_radius

        Args:
            num_cells: Total cells at data level
            masking_strategy_config: Config for student cone, must contain:
                - 'rate': Fraction of sphere for student cone
                - 'hl_mask': HEALPix level for cone generation
            teacher_center_cell: HEALPix cell index of teacher cone center
            teacher_rate: Teacher's area fraction (to compute teacher radius)
            teacher_hl_mask: HEALPix level of teacher center cell

        Returns:
            Tuple of (student_mask, student_center_cell)

        Mathematical Details:
            - Teacher radius: theta_T = arccos(1 - 2*teacher_rate)
            - Student radius: theta_S = arccos(1 - 2*student_rate)
            - Minimum distance: d_min = theta_T + theta_S
            - Maximum distance: pi (opposite side of sphere)
            - Actual distance: sampled from [d_min, pi]
        """
        # Get student parameters
        mask_level = masking_strategy_config.get("hl_mask", 0)
        student_rate = masking_strategy_config.get("rate", 0.5)
        hp_mask = self._get_hp_obj(mask_level)

        # Calculate radii using spherical cap formula
        teacher_radius_rad = np.arccos(1 - 2 * teacher_rate)
        student_radius_rad = np.arccos(1 - 2 * student_rate)

        # Minimum distance for separation
        min_distance_rad = teacher_radius_rad + student_radius_rad
        max_distance_rad = np.pi  # Opposite side of sphere

        # Validate constraint is satisfiable
        if min_distance_rad > np.pi:
            raise ValueError(
                f"separated_cone requires teacher_rate + student_rate such that cones can fit. "
                f"Combined radii ({np.degrees(min_distance_rad):.1f}deg) > 180deg"
            )

        # Sample distance (default: random within valid range)
        distance_mode = masking_strategy_config.get("separated_distance_mode", "random")
        if distance_mode == "random":
            # Uniform in [min_distance, pi]
            distance_rad = self.rng.uniform(min_distance_rad, max_distance_rad)
        elif distance_mode == "touching":
            distance_rad = min_distance_rad
        elif distance_mode == "opposite":
            distance_rad = max_distance_rad
        else:
            # Fixed fraction between min and max
            distance_fraction = masking_strategy_config.get("separated_distance_fraction", 0.5)
            distance_rad = min_distance_rad + distance_fraction * (
                max_distance_rad - min_distance_rad
            )

        # Get teacher center coordinates
        hp_teacher = self._get_hp_obj(teacher_hl_mask)
        teacher_lon, teacher_lat = hp_teacher.healpix_to_lonlat(teacher_center_cell)

        # Random azimuth direction
        azimuth_rad = self.rng.uniform(0, 2 * np.pi)

        # Compute student center using great circle navigation
        student_lon_rad, student_lat_rad = self._get_destination_latlon(
            teacher_lon.to_value(u.rad),
            teacher_lat.to_value(u.rad),
            distance_rad,
            azimuth_rad,
        )

        # Create student cone using tree-based search
        selected_parent_cells = hp_mask.cone_search_lonlat(
            student_lon_rad * u.rad,
            student_lat_rad * u.rad,
            radius=student_radius_rad * u.rad,
        )

        # Get student center cell index
        student_center_cell = int(
            hp_mask.lonlat_to_healpix(student_lon_rad * u.rad, student_lat_rad * u.rad)
        )

        # Build mask using NESTED ordering properties
        mask_array = np.zeros(num_cells, dtype=bool)
        level_diff = self.healpix_level_data - mask_level

        if level_diff > 0:
            num_children_per_parent = 4**level_diff
            for parent_idx in selected_parent_cells:
                start_idx = parent_idx * num_children_per_parent
                end_idx = start_idx + num_children_per_parent
                mask_array[start_idx:end_idx] = True
        else:
            mask_array[selected_parent_cells] = True

        return torch.from_numpy(mask_array), student_center_cell
