# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import ViewMetadata
from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
)
from weathergen.datasets.view_builder import build_views_for_stream


class TokenizerMasking(Tokenizer):
    def __init__(self, healpix_level: int, masker: Masker):
        super().__init__(healpix_level)
        self.masker = masker
        # cache last built view metadata per stream invocation (optional downstream use)
        self._last_view_metadata: list[ViewMetadata] | None = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after epoch to ensure proper randomization
        """
        self.masker.reset_rng(rng)
        self.rng = rng

    def batchify_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        idxs_cells_data,
        time_win: tuple,
        keep_mask: torch.Tensor | None = None,
    ):
        token_size = stream_info["token_size"]
        stream_id = stream_info["stream_id"]
        assert token_size is not None, "stream did not specify token_size"
        is_diagnostic = stream_info.get("diagnostic", False)

        # return empty if there is no data or we are in diagnostic mode
        if is_diagnostic or rdata.data.shape[1] == 0 or len(rdata.data) < 2:
            source_tokens_cells = [torch.tensor([])]
            source_tokens_lens = torch.zeros([self.num_healpix_cells_source], dtype=torch.int32)
            mask_state = {
                "strategy": self.masker.current_strategy,
                "mask_tokens": None,
                "mask_channels": None,
            }
            return (source_tokens_cells, source_tokens_lens, mask_state)

        # # create tokenization index
        # tok = tokenize_spacetime if stream_info.get("tokenize_spacetime", False) else tokenize_space
        # idxs_cells, idxs_cells_lens = tok(rdata, token_size, self.hl_source, pad_tokens=True)
        (idxs_cells, idxs_cells_lens) = idxs_cells_data

        # select strategy from XXX depending on stream and if student or teacher

        # Optional per-cell keep_mask (boolean) converts to numpy for Masker override.
        if keep_mask is not None:
            keep_np = keep_mask.cpu().numpy().astype(bool)
            (mask_tokens, mask_channels) = self.masker.mask_source_idxs(
                stream_info, idxs_cells, idxs_cells_lens, rdata, keep_mask=keep_np
            )
        else:
            (mask_tokens, mask_channels) = self.masker.mask_source_idxs(
                stream_info, idxs_cells, idxs_cells_lens, rdata
            )

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_id,
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
        )

        # capture per-view mask state to later produce consistent targets
        mask_state = {
            "strategy": self.masker.current_strategy,
            "mask_tokens": mask_tokens,
            "mask_channels": mask_channels,
        }
        self.mask_state = mask_state

        return (source_tokens_cells, source_tokens_lens, mask_state)

    # batchify_target_for_view now unified into batchify_target via optional mask_state

    def batchify_target(
        self,
        stream_info: dict,
        sampling_rate_target: float,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        mask_state: dict | None = None,
    ):
        token_size = stream_info["token_size"]

        # create tokenization index
        # tok = tokenize_spacetime if stream_info.get("tokenize_spacetime", False) else tokenize_space
        # idxs_cells, idxs_cells_lens = tok(rdata, token_size, self.hl_source, pad_tokens=False)
        (idxs_cells, idxs_cells_lens) = token_data

        mask_state = self.mask_state

        # Apply per-view mask state if provided
        if mask_state is not None:
            self.masker.current_strategy = mask_state.get("strategy", self.masker.masking_strategy)
            self.masker.mask_tokens = mask_state.get("mask_tokens")
            self.masker.mask_channels = mask_state.get("mask_channels")

        (mask_tokens, mask_channels, idxs_ord_inv) = self.masker.mask_targets_idxs(
            stream_info, idxs_cells, idxs_cells_lens, rdata
        )

        data, datetimes, coords, coords_local, coords_per_cell = tokenize_apply_mask_target(
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
        )

        # TODO, TODO, TODO: max_num_targets
        # max_num_targets = stream_info.get("max_num_targets", -1)

        return (data, datetimes, coords, coords_local, coords_per_cell, idxs_ord_inv)

    # ------------------------------------------------------------------
    # Per-stream view construction (teacher + students) for student-teacher
    # ------------------------------------------------------------------
    def build_stream_views(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        time_win: tuple,
        training_cfg: dict | None = None,
    ):
        """Construct teacher and student views for a single stream.

        Parameters
        ----------
        stream_info : dict
            Stream configuration dictionary.
        rdata : IOReaderData
            Combined reader data for this stream.
        time_win : tuple
            (start, end) datetime window.
        training_cfg : dict | None
            cf.training_config section; if absent or mode != 'student_teacher', fallback to single view.

        Returns
        -------
        teacher : tuple | None
            (tokens_cells, tokens_lens) for teacher or None when not student_teacher.
        students : list
            List of (tokens_cells, tokens_lens) for each student view (or single masking view).
        view_metadata : list[ViewMetadata] | None
            Metadata for teacher + students when in student_teacher mode.
        """
        if training_cfg is None or training_cfg.get("training_mode") != "student_teacher":
            # Standard masking path: single view only (treated as 'student' for uniformity)
            scells, slens, _mask_state = self.batchify_source(stream_info, rdata, time_win)
            self._last_view_metadata = None
            return None, [(scells, slens, _mask_state)], None

        teacher_cfg = training_cfg.get("teacher_model_input", {})
        student_cfg = training_cfg.get("model_input", {})
        relationship = student_cfg.get("relationship", "subset")

        num_cells = self.num_healpix_cells_source
        teacher_keep_mask, student_keep_masks, view_meta = build_views_for_stream(
            self.masker, num_cells, teacher_cfg, student_cfg, relationship
        )

        # Convert keep masks to torch tensors for downstream masking override
        teacher_keep_mask_t = torch.from_numpy(teacher_keep_mask)
        student_keep_masks_t = [torch.from_numpy(m) for m in student_keep_masks]

        # Teacher tokens
        t_cells, t_lens, t_mask_state = self.batchify_source(
            stream_info, rdata, time_win, keep_mask=teacher_keep_mask_t
        )
        # Student tokens
        student_tokens = [
            self.batchify_source(stream_info, rdata, time_win, keep_mask=km)
            for km in student_keep_masks_t
        ]
        # add mask_state inside each tuple
        student_tokens = [(cells, lens, mstate) for (cells, lens, mstate) in student_tokens]

        self._last_view_metadata = view_meta
        return (t_cells, t_lens, t_mask_state), student_tokens, view_meta

    def sample_tensors_uniform_vectorized(
        self, tensor_list: list, lengths: list, max_total_points: int
    ):
        """
        This function randomly selects tensors up to a maximum number of total points

        tensor_list: List[torch.tensor] the list to select from
        lengths: List[int] the length of each tensor in tensor_list
        max_total_points: the maximum number of total points to sample from
        """
        if not tensor_list:
            return [], 0

        # Create random permutation
        perm = self.rng.permutation(len(tensor_list))

        # Vectorized cumulative sum
        cumsum = torch.cumsum(lengths[perm], dim=0)

        # Find cutoff point
        valid_mask = cumsum <= max_total_points
        if not valid_mask.any():
            return [], 0

        num_selected = valid_mask.sum().item()
        perm = torch.tensor(perm)
        selected_indices = perm[:num_selected]
        selected_indices = torch.zeros_like(perm).scatter(0, selected_indices, 1)

        selected_tensors = [
            t if mask.item() == 1 else t[:0]
            for t, mask in zip(tensor_list, selected_indices, strict=False)
        ]

        return selected_tensors
