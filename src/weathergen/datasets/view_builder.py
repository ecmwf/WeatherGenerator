import numpy as np
from typing import Tuple, List
from weathergen.datasets.masking import Masker
from weathergen.datasets.inputs_metadata import ViewMetadata


def build_views_for_stream(
    masker: Masker,
    num_cells: int,
    teacher_cfg: dict,
    student_cfg: dict,
    relationship: str = "subset",
) -> Tuple[np.ndarray, List[np.ndarray], List[ViewMetadata]]:
    """
    
    Per-stream view construction: teacher + N student keep masks.

    Parameters
    ----------
    masker : Masker
        Instance providing RNG and healpix-level info.
    num_cells : int
        Number of healpix cells at data level.
    teacher_cfg : dict
        Config: {strategy, rate|keep_m, hl_mask, masking_strategy_config, rate_sampling}.
    student_cfg : dict
        Config: {masking_strategy, rate, num_views, hl_mask, masking_strategy_config, rate_sampling}.
    relationship : str
        One of {'subset','disjoint','independent'}. Determines derivation of student masks.

    Returns
    -------
    teacher_keep_mask : np.ndarray
        Boolean keep mask for teacher view.
    student_keep_masks : list[np.ndarray]
        Boolean keep masks for each student view.
    metadata : list[ViewMetadata]
        Metadata objects (teacher first, then students).
    
    """
    strat_teacher = teacher_cfg.get("strategy", "random")
    rate_teacher = teacher_cfg.get("rate")
    t_cfg_extra = teacher_cfg.get("masking_strategy_config")

    teacher_keep_mask = masker.generate_cell_keep_mask(
        num_cells=num_cells,
        strategy=strat_teacher,
        rate=rate_teacher,
        masking_strategy_config=t_cfg_extra,
    )

    # Student base masks
    num_views = student_cfg.get("num_views", 1)
    strat_student = student_cfg.get("masking_strategy", student_cfg.get("strategy", "random"))
    rate_student = student_cfg.get("rate")
    s_cfg_extra = student_cfg.get("masking_strategy_config")

    student_keep_masks: List[np.ndarray] = []
    for v in range(num_views):
        base = masker.generate_cell_keep_mask(
            num_cells=num_cells,
            strategy=strat_student,
            rate=rate_student,
            masking_strategy_config=s_cfg_extra,
        )
        if relationship == "subset":
            keep = base & teacher_keep_mask
        elif relationship == "disjoint":
            keep = base & (~teacher_keep_mask)
        else:  # independent
            keep = base
        student_keep_masks.append(keep)

    metadata: List[ViewMetadata] = []
    metadata.append(
        ViewMetadata(
            view_id="teacher_global",
            keep_mask=teacher_keep_mask,
            strategy=strat_teacher,
            healpix_level=masker.healpix_level_data,
            rate=rate_teacher,
            parent_view_id=None,
        )
    )
    for i, m in enumerate(student_keep_masks):
        metadata.append(
            ViewMetadata(
                view_id=f"student_local_{i}",
                keep_mask=m,
                strategy=strat_student,
                healpix_level=masker.healpix_level_data,
                rate=rate_student,
                parent_view_id="teacher_global",
            )
        )

    return teacher_keep_mask, student_keep_masks, metadata
