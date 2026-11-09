import logging

logger = logging.getLogger(__name__)


def apply_curriculum(cf) -> None:
    """
    Applies the curriculum to the configuration if `healpix_curriculum` is present.
    Determines the current healpix_level based on the cumulative steps, and
    updates `cf.healpix_level` and `cf.streams_directory` accordingly.
    """
    if cf.get("healpix_curriculum"):
        istep = cf.get("general", {}).get("istep", 0)
        cumulative = 0
        current_hl = None
        unique_curr = {int(hl): steps for hl, steps in cf.healpix_curriculum.items()}
        for hl in sorted(unique_curr.keys()):
            cumulative += unique_curr[hl]
            current_hl = hl
            if istep < cumulative:
                break
        cf.healpix_level = current_hl

        if cf.get("curriculum_streams"):
            # Support both integer and string keys in the yaml
            cf.streams_directory = cf.curriculum_streams.get(
                current_hl
            ) or cf.curriculum_streams.get(str(current_hl))

        # Pre-calculate the exact istep when this curriculum stage should exit
        cf._curriculum_exit_step = None
        max_hl = max(unique_curr.keys())
        if current_hl < max_hl:
            cf._curriculum_exit_step = sum(
                steps for hl, steps in unique_curr.items() if hl <= current_hl
            )
