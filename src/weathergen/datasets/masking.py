import copy
import logging
import warnings
from dataclasses import dataclass, field

import astropy_healpix as hp
import numpy as np
import omegaconf
import torch
from numpy.typing import NDArray

from weathergen.datasets.batch import SampleMetaData

logger = logging.getLogger(__name__)

# Earth radius in km (used for correlation length calculations)
EARTH_RADIUS_KM = 6371.0


@dataclass
class ChannelMaskingConfig:
    """
    Configuration for per-channel spatial masking based on autocorrelation.

    Attributes
    ----------
    autocorr : dict[str, dict]
        Per-channel autocorrelation info. Each entry maps channel name to
        {"space_km": float, "time_h": float} (spatial/temporal correlation scales).
    multiplier : float
        Scale factor for mapping correlation length to mask size (default: 1.0).
    hl_mask_min : int
        Minimum HEALPix level (prevents hemisphere-scale masks).
    hl_mask_max : int
        Maximum HEALPix level (prevents sub-grid masks).
    default_hl_mask : int
        Default HEALPix level for channels not in autocorr dict.
    enabled : bool
        Whether per-channel masking is enabled.

    Examples
    --------
    >>> config = ChannelMaskingConfig(
    ...     autocorr={
    ...         "z_500": {"space_km": 4000},
    ...         "tp": {"space_km": 177},
    ...     },
    ...     enabled=True,
    ... )
    >>> config.get_hl_mask("z_500")  # Large-scale variable
    1
    >>> config.get_hl_mask("tp")     # Small-scale variable
    5
    """

    autocorr: dict = field(default_factory=dict)
    multiplier: float = 1.0
    hl_mask_min: int = 1
    hl_mask_max: int = 5
    default_hl_mask: int = 3
    enabled: bool = False

    def get_hl_mask(self, channel_name: str) -> int:
        """
        Get HEALPix masking level for a channel.

        Parameters
        ----------
        channel_name : str
            Name of the channel (e.g., "z_500", "tp").

        Returns
        -------
        int
            HEALPix level for masking this channel.
        """
        if not self.enabled or channel_name not in self.autocorr:
            return self.default_hl_mask

        ch_info = self.autocorr[channel_name]
        l_corr_km = ch_info.get("space_km", None)

        if l_corr_km is None:
            return self.default_hl_mask

        return correlation_length_to_hl_mask(
            l_corr_km,
            multiplier=self.multiplier,
            hl_min=self.hl_mask_min,
            hl_max=self.hl_mask_max,
        )

    @classmethod
    def from_config(cls, cfg: dict | None) -> "ChannelMaskingConfig":
        """
        Create from a config dict (e.g., from YAML).

        Parameters
        ----------
        cfg : dict | None
            Config dict with optional keys: enabled, autocorr, multiplier, etc.

        Returns
        -------
        ChannelMaskingConfig
            Configured instance.
        """
        if cfg is None:
            return cls()

        # Handle OmegaConf -> dict conversion
        if hasattr(cfg, "to_container"):
            cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

        return cls(
            autocorr=cfg.get("autocorr", {}),
            multiplier=cfg.get("multiplier", 1.0),
            hl_mask_min=cfg.get("hl_mask_min", 1),
            hl_mask_max=cfg.get("hl_mask_max", 5),
            default_hl_mask=cfg.get("default_hl_mask", 3),
            enabled=cfg.get("enabled", False),
        )


def correlation_length_to_hl_mask(
    l_corr_km: float,
    multiplier: float = 1.0,
    hl_min: int = 1,
    hl_max: int = 5,
) -> int:
    """
    Map spatial correlation length to HEALPix masking level.

    Finds the finest HEALPix level where the cell size is still larger
    than the correlation length scaled by the multiplier. This ensures
    mask regions are large enough to prevent trivial interpolation
    based on the variable's spatial autocorrelation structure.

    Parameters
    ----------
    l_corr_km : float
        Spatial correlation length in km (e-folding distance).
        Typically computed from analysis of the variable's spatial
        autocorrelation function using scripts/compute_autocorr.py.
    multiplier : float
        Scale factor applied to correlation length (default: 1.0).
        With multiplier=1.0, mask cell size >= correlation length.
        Higher values create coarser masks (larger cells).
    hl_min : int
        Minimum HEALPix level (default: 1, prevents hemisphere-scale masks).
        Level 1 has 48 cells (~3260 km each).
    hl_max : int
        Maximum HEALPix level (default: 5, prevents sub-grid masks).
        Level 5 has 12288 cells (~204 km each).

    Returns
    -------
    int
        HEALPix level for masking. Lower levels = larger masks.

    Examples
    --------
    >>> correlation_length_to_hl_mask(4000)  # z_500 (L_corr ~ 4000 km)
    1
    >>> correlation_length_to_hl_mask(1200)  # 10u wind (L_corr ~ 1200 km)
    2
    >>> correlation_length_to_hl_mask(177)   # precipitation (L_corr ~ 177 km)
    5

    Notes
    -----
    HEALPix cell sizes (approximate):
        Level 0: ~6520 km
        Level 1: ~3260 km
        Level 2: ~1630 km
        Level 3: ~815 km
        Level 4: ~408 km
        Level 5: ~204 km
        Level 6: ~102 km

    Reference correlation lengths from ERA5 O96 analysis:
        z_500:  ~4000 km -> Level 1
        t_850:  ~4000 km -> Level 1
        2t:     ~3770 km -> Level 1
        u_50:   ~3160 km -> Level 1
        q_850:  ~3100 km -> Level 1
        v_50:   ~1280 km -> Level 2
        u_1000: ~1220 km -> Level 2
        10u:    ~1200 km -> Level 2
        10v:    ~750 km  -> Level 3
        tp:     ~180 km  -> Level 5
    """
    target_size = l_corr_km * multiplier

    # Find finest level (highest number) where cell_size > target_size
    for hl in range(hl_max, hl_min - 1, -1):
        cell_size = _healpix_cell_size_km(hl)
        if cell_size > target_size:
            return hl

    # If no level is fine enough, return the minimum (coarsest allowed)
    return hl_min


def _healpix_cell_size_km(healpix_level: int) -> float:
    """
    Compute approximate HEALPix cell size in km.

    Parameters
    ----------
    healpix_level : int
        HEALPix level (nside = 2^level).

    Returns
    -------
    float
        Approximate cell diameter in km.
    """
    n_cells = 12 * (4**healpix_level)
    cell_area_km2 = (4 * np.pi * EARTH_RADIUS_KM**2) / n_cells
    # Approximate cell size as sqrt of area (diameter of equivalent circle)
    return np.sqrt(cell_area_km2)


def time_corr_to_block_size(
    t_corr_hours: float,
    period_hours: float,
    multiplier: float = 1.5,
    min_block: int = 1,
    max_block: int = 8,
) -> int:
    """
    Map temporal correlation scale to time blocking size for tube masking.

    Computes the number of time steps to mask as a contiguous block
    to prevent trivial temporal interpolation.

    Parameters
    ----------
    t_corr_hours : float
        Temporal correlation scale in hours (e-folding time).
    period_hours : float
        Time step between samples in hours (e.g., 6 for 6-hourly data).
    multiplier : float
        Scale factor for decorrelation (default: 1.5).
    min_block : int
        Minimum block size in time steps.
    max_block : int
        Maximum block size in time steps.

    Returns
    -------
    int
        Number of time steps to mask as a block.

    Examples
    --------
    >>> time_corr_to_block_size(48, 6)  # 48h correlation, 6h data
    2
    >>> time_corr_to_block_size(12, 6)  # 12h correlation, 6h data
    1
    """
    t_mask_hours = multiplier * t_corr_hours
    block_size = int(np.ceil(t_mask_hours / period_hours))
    return max(min_block, min(max_block, block_size))


def generate_tube_mask(
    spatial_mask: np.ndarray | torch.Tensor,
    num_timesteps: int,
) -> np.ndarray:
    """
    Generate a tube mask by replicating the same spatial mask across all timesteps.

    Tube masking ensures that the same spatial regions are masked/kept across
    all timesteps in a window, preventing the model from using temporal
    interpolation to "cheat" on the prediction task.

    Parameters
    ----------
    spatial_mask : np.ndarray | torch.Tensor
        1D boolean mask of shape [num_cells] where True = keep, False = mask.
    num_timesteps : int
        Number of timesteps in the time window.

    Returns
    -------
    np.ndarray
        2D boolean mask of shape [num_timesteps, num_cells].

    Examples
    --------
    >>> spatial_mask = np.array([True, False, True, False])
    >>> tube_mask = generate_tube_mask(spatial_mask, 3)
    >>> tube_mask.shape
    (3, 4)
    >>> np.all(tube_mask[0] == tube_mask[1])  # Same mask at all times
    True
    """
    if isinstance(spatial_mask, torch.Tensor):
        spatial_mask = spatial_mask.numpy()

    # Replicate spatial mask across all timesteps
    return np.tile(spatial_mask, (num_timesteps, 1))


def apply_tube_mask_to_obs(
    spatial_mask: np.ndarray | torch.Tensor,
    obs_cell_indices: np.ndarray,
) -> np.ndarray:
    """
    Apply tube masking to irregular observations.

    For irregular observation data (e.g., from DataReaderObs), each observation
    belongs to a specific HEALPix cell. This function applies the spatial mask
    to observations based on which cell they belong to.

    Parameters
    ----------
    spatial_mask : np.ndarray | torch.Tensor
        1D boolean mask of shape [num_cells] where True = keep, False = mask.
    obs_cell_indices : np.ndarray
        1D array of shape [num_obs] containing the cell index for each observation.

    Returns
    -------
    np.ndarray
        1D boolean mask of shape [num_obs] where True = keep observation.

    Examples
    --------
    >>> spatial_mask = np.array([True, False, True])  # Keep cells 0 and 2
    >>> obs_cells = np.array([0, 0, 1, 2, 1, 2])  # 6 observations in 3 cells
    >>> obs_mask = apply_tube_mask_to_obs(spatial_mask, obs_cells)
    >>> obs_mask
    array([ True,  True, False,  True, False,  True])
    """
    if isinstance(spatial_mask, torch.Tensor):
        spatial_mask = spatial_mask.numpy()

    return spatial_mask[obs_cell_indices]


class MaskData:
    masks: list[np.typing.NDArray] = []
    metadata: list[SampleMetaData] = []

    def __init__(self):
        self.masks = []
        self.metadata = []

    def __len__(self):
        return len(self.masks)

    def add_mask(
        self,
        mask,
        params,
        cfg,
        losses,
        idx,
        correspondence,
        relationship,
        temporal_strategy: str | None = None,
        channel_masks: dict[str, np.ndarray] | None = None,
    ):
        """
        Add a mask with associated metadata.

        Parameters
        ----------
        mask : np.ndarray | torch.Tensor
            Boolean mask of shape [num_cells] where True = keep.
        params : dict
            Additional parameters describing the mask.
        cfg : dict
            Configuration used to generate the mask.
        losses : list
            Loss functions associated with this mask.
        idx : int
            Index of this mask in the collection.
        correspondence : list
            Source-target correspondence information.
        relationship : str | None
            Relationship type (e.g., "complement", "subset").
        temporal_strategy : str | None
            Temporal masking strategy. Options:
            - None: No temporal consistency (independent masks per timestep)
            - "tube": Same spatial mask applied to all timesteps
            - "block": Block masking in time (future extension)
        channel_masks : dict[str, np.ndarray] | None
            Per-channel spatial masks. Keys are channel names, values are
            boolean masks of shape [num_cells]. If provided, these override
            the uniform spatial mask for per-variable masking.
        """
        global_params = {
            "idx": idx,
            "correspondence": correspondence,
            "loss": losses,
            "relationship": relationship,
        }

        # Add temporal masking info if specified
        if temporal_strategy is not None:
            global_params["temporal_strategy"] = temporal_strategy

        # Add per-channel masks if specified
        if channel_masks:
            global_params["channel_masks"] = channel_masks

        self.masks += [mask]
        self.metadata += [
            SampleMetaData(
                params={**cfg, **params},
                mask=mask,
                global_params=global_params,
            )
        ]

    def get_mask(self, idx: int) -> np.typing.NDArray:
        return self.masks[idx]

    def get_channel_masks(self, idx: int) -> dict[str, np.ndarray] | None:
        """Get the per-channel masks for a given mask index."""
        global_params = self.metadata[idx].global_params
        if global_params is None:
            return None
        return global_params.get("channel_masks", None)

    def get_temporal_strategy(self, idx: int) -> str | None:
        """Get the temporal masking strategy for a given mask index."""
        global_params = self.metadata[idx].global_params
        if global_params is None:
            return None
        return global_params.get("temporal_strategy", None)


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

    def __init__(self, healpix_level):
        self.rng = None

        self.mask_value = 0.0
        self.dim_time_enc = 6

        # number of healpix cells
        self.healpix_level_data = healpix_level
        self.num_healpix_cells = 12 * (4**healpix_level)

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
        channel_list: list[str] | None = None,
        channel_masking_config: ChannelMaskingConfig | None = None,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Construct teacher/student keep masks for a stream.

        Parameters
        ----------
        training_mode : str
            Training mode identifier.
        num_cells : int
            Number of HEALPix cells at data level.
        stage_cfg : dict
            Stage configuration with target_input, model_input, losses, etc.
        channel_list : list[str] | None
            Optional list of channel names for per-channel masking.
        channel_masking_config : ChannelMaskingConfig | None
            Optional per-channel masking configuration.

        Returns
        -------
        tuple
            (target_masks, source_masks, source_target_mapping)
        """

        # Generate per-channel masks if config provided
        channel_masks = {}
        if channel_list is not None and channel_masking_config is not None:
            # Get keep_rate from first source config (or default)
            source_cfgs = stage_cfg.get("model_input", {})
            keep_rate = 0.5
            for _, src_cfg in source_cfgs.items():
                strat_cfg = src_cfg.get("masking_strategy_config", {})
                if "rate" in strat_cfg:
                    keep_rate = strat_cfg["rate"]
                    break

            channel_masks = self.generate_channel_masks(
                channel_list, channel_masking_config, keep_rate
            )

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

                # Extract temporal strategy from config (default: "tube" for temporal consistency)
                masking_strategy_config = target_cfg.get("masking_strategy_config", {})
                temporal_strategy = masking_strategy_config.get("temporal_strategy", None)

                # add
                target_masks.add_mask(
                    target_mask,
                    mask_params,
                    target_cfg,
                    losses,
                    i_target,
                    corr,
                    None,
                    temporal_strategy=temporal_strategy,
                    channel_masks=channel_masks if channel_masks else None,
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

                source_mask, mask_params = self._get_mask(
                    num_cells=num_cells,
                    strategy=source_cfg.get("masking_strategy"),
                    masking_strategy_config=masking_config,
                    target_relationship_mask=(relationship, target_masks.get_mask(target_idx)),
                )
                corr = target_idx

                # Extract temporal strategy from config
                temporal_strategy = masking_config.get("temporal_strategy", None)

                source_masks.add_mask(
                    source_mask,
                    mask_params,
                    source_cfg,
                    losses,
                    i_source,
                    corr,
                    relationship,
                    temporal_strategy=temporal_strategy,
                    channel_masks=channel_masks if channel_masks else None,
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
            return mask, {}

        # get mask
        mask, params = self._generate_cell_mask(num_cells, strategy, masking_strategy_config)

        # handle cases where mask needs to be combined with target_mask
        # without the assert we can fail silently
        if relationship == "subset":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
            mask = mask & target_mask
        elif relationship == "disjoint":
            assert target_mask is not None, (
                "relationship: {relationship} incompatible with target_mask None"
            )
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

        Returns
        -------
        np.ndarray
            Boolean array of shape [num_cells] where True indicates the cell is kept.
        """

        # params describing the masking
        masking_params = {}

        assert num_cells == self.num_healpix_cells, (
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

                # Use standard spatial selection
                mask = self._select_spatially_contiguous_cells(
                    healpix_level=hl_mask,
                    num_cells=num_cells,
                    num_cells_to_select=num_parents_to_keep,
                    num_children_per_parent=num_children_per_parent,
                    center_cell=None,
                    method=method,
                )

        else:
            raise NotImplementedError(
                f"Cell selection strategy '{strategy}' not supported for keep mask generation."
            )

        mask = to_bool_tensor(mask)

        return (mask, masking_params)

    def _select_spatially_contiguous_cells(
        self,
        healpix_level: int,
        num_cells: int,
        num_cells_to_select: int,
        num_children_per_parent: int,
        center_cell: int | None = None,
        method: str = "geodesic_disk",
    ) -> NDArray:
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
            Array of selected cell indices forming a spatially contiguous region

        Examples:
            # Independent crop
            crop1 = _select_spatially_contiguous_cells(0, 9, method="geodesic_disk")
        """

        num_total_cells = 12 * (4**healpix_level)
        nside = 2**healpix_level

        assert num_cells_to_select <= num_total_cells

        # Random starting point. Note we may want overlap here
        # for now we basically control with chosen masking rates
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

        return mask

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

    def generate_channel_masks(
        self,
        channel_list: list[str],
        channel_masking_config: ChannelMaskingConfig,
        keep_rate: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """
        Generate per-channel spatial masks based on autocorrelation.

        Each channel gets a mask at the HEALPix level appropriate for its
        spatial correlation length. Channels with the same hl_mask share
        the same spatial mask pattern.

        Parameters
        ----------
        channel_list : list[str]
            List of channel names in order (e.g., ["z_500", "t_850", "tp"]).
        channel_masking_config : ChannelMaskingConfig
            Configuration specifying autocorrelation and mapping parameters.
        keep_rate : float
            Fraction of cells to keep unmasked (0 to 1).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from channel name to boolean mask [num_cells].
            True = keep, False = mask.

        Examples
        --------
        >>> config = ChannelMaskingConfig(
        ...     autocorr={"z_500": {"space_km": 4000}, "tp": {"space_km": 177}},
        ...     enabled=True,
        ... )
        >>> masks = masker.generate_channel_masks(["z_500", "tp"], config, keep_rate=0.5)
        >>> masks["z_500"].shape  # Coarser mask (level 1)
        (num_cells,)
        >>> masks["tp"].shape     # Finer mask (level 5)
        (num_cells,)
        """
        if not channel_masking_config.enabled:
            # Return empty dict if not enabled - caller uses default spatial mask
            return {}

        num_cells = self.num_healpix_cells
        hl_data = self.healpix_level_data

        # Group channels by their HEALPix masking level
        hl_to_channels: dict[int, list[str]] = {}
        for ch in channel_list:
            hl = channel_masking_config.get_hl_mask(ch)
            # Cap hl_mask at data level - 1 (required by _prepare_healpix_based_masking)
            hl = min(hl, hl_data - 1)
            if hl not in hl_to_channels:
                hl_to_channels[hl] = []
            hl_to_channels[hl].append(ch)

        # Generate one mask per unique hl_mask level
        hl_masks: dict[int, np.ndarray] = {}
        for hl in hl_to_channels:
            mask, _ = self._generate_cell_mask(
                num_cells,
                strategy="healpix",
                masking_strategy_config={"hl_mask": hl, "rate": keep_rate},
            )
            # Convert to numpy if torch
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            hl_masks[hl] = mask

        # Map each channel to its mask
        channel_masks = {}
        for ch in channel_list:
            hl = channel_masking_config.get_hl_mask(ch)
            # Cap hl_mask at data level - 1 (must match grouping above)
            hl = min(hl, hl_data - 1)
            channel_masks[ch] = hl_masks[hl]

        return channel_masks
