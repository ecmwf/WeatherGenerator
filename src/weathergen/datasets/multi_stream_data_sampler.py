# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import pathlib

import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    TimeWindowHandler,
    TIndex,
    str_to_datetime64,
)
from weathergen.datasets.data_reader_fesom import DataReaderFesom
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.icon_dataset import IconDataset
from weathergen.datasets.masking import Masker
from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.datasets.tokenizer_forecast import TokenizerForecast
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.views import ModelBatch, ViewMetadata
from weathergen.datasets.utils import (
    compute_idxs_predict,
    compute_offsets_scatter_embed,
    compute_source_cell_lens,
)
from weathergen.utils.distributed import is_root
from weathergen.utils.train_logger import Stage

type AnyDataReader = DataReaderBase | DataReaderAnemoi | DataReaderObs

logger = logging.getLogger(__name__)


def readerdata_to_torch(rdata: IOReaderData) -> IOReaderData:
    """
    Convert data, coords, and geoinfos to torch tensor.
    Creates a new instance to avoid mutating the original numpy data.
    """
    return IOReaderData(
        coords=torch.as_tensor(rdata.coords),
        geoinfos=torch.as_tensor(rdata.geoinfos),
        data=torch.as_tensor(rdata.data),
        datetimes=rdata.datetimes,  # keep numpy
    )


def collect_datasources(stream_datasets: list, idx: int, type: str) -> IOReaderData:
    """
    Utility function to collect all sources / targets from streams list
    """
    rdatas = []

    for ds in stream_datasets:
        if type == "source":
            get_reader_data = ds.get_source
            normalize_channels = ds.normalize_source_channels
        elif type == "target":
            get_reader_data = ds.get_target
            normalize_channels = ds.normalize_target_channels
        else:
            assert False, "invalid value for argument `type`"

        # get source (of potentially multi-step length)
        rdata = get_reader_data(idx).remove_nan_coords()
        rdata.data = normalize_channels(rdata.data)
        rdata.geoinfos = ds.normalize_geoinfos(rdata.geoinfos)
        rdatas += [rdata]

    return IOReaderData.combine(rdatas)


class MultiStreamDataSampler(torch.utils.data.IterableDataset):
    """
    Multi-stream data sampler supporting three training modes:
      - "forecast": standard forecasting (no masking)
      - "masking": MAE-style masked autoencoding
      - "student_teacher": JEPA-style multi-view training
    """

    def __init__(
        self,
        cf,
        start_date_,
        end_date_,
        batch_size,
        samples_per_epoch,
        stage: Stage,
        shuffle=True,
    ):
        super(MultiStreamDataSampler, self).__init__()

        start_date = str_to_datetime64(start_date_)
        end_date = str_to_datetime64(end_date_)

        assert end_date > start_date, (end_date, start_date)

        self.mask_value = 0.0
        self._stage = stage

        self.len_hrs: int = cf.len_hrs
        self.step_hrs: int = cf.step_hrs
        self.time_window_handler = TimeWindowHandler(start_date, end_date, cf.len_hrs, cf.step_hrs)
        if is_root():
            logger.info(
                f"Time window handler: start={start_date}, end={end_date},"
                f"len_hrs={cf.len_hrs}, step_hrs={cf.step_hrs}"
            )

        self.forecast_offset = cf.forecast_offset
        self.forecast_delta_hrs = (
            cf.forecast_delta_hrs if cf.forecast_delta_hrs > 0 else self.len_hrs
        )
        assert self.forecast_delta_hrs == self.len_hrs, "Only supported option at the moment"
        self.forecast_steps = np.array(
            [cf.forecast_steps] if isinstance(cf.forecast_steps, int) else cf.forecast_steps
        )
        if cf.forecast_policy is not None:
            if self.forecast_steps.max() == 0 and is_root():
                logger.warning("forecast policy is not None but number of forecast steps is 0.")
        self.forecast_policy = cf.forecast_policy

        self.len = 100000000

        # Initialize datasets for each stream
        self.streams_datasets: list[list[AnyDataReader]] = []
        for _, stream_info in enumerate(cf.streams):
            self.streams_datasets.append([])

            for fname in stream_info["filenames"]:
                kwargs = {
                    "tw_handler": self.time_window_handler,
                    "stream_info": stream_info,
                }
                dataset: type[AnyDataReader] | None = None
                match stream_info["type"]:
                    case "obs":
                        dataset = DataReaderObs
                        datapath = cf.data_path_obs
                    case "anemoi":
                        dataset = DataReaderAnemoi
                        datapath = cf.data_path_anemoi
                    case "fesom":
                        dataset = DataReaderFesom
                        datapath = cf.data_path_fesom
                    case "icon":
                        dataset = IconDataset
                        datapath = cf.data_path_icon
                    case _:
                        msg = f"Unsupported stream type {stream_info['type']}"
                        f"for stream name '{stream_info['name']}'."
                        raise ValueError(msg)

                datapath = pathlib.Path(datapath)
                fname = pathlib.Path(fname)
                if fname.exists():
                    filename = fname
                else:
                    filename = pathlib.Path(datapath) / fname
                    if not filename.exists():
                        msg = (
                            f"Did not find input data for {stream_info['type']} "
                            f"stream '{stream_info['name']}': {filename}."
                        )
                        raise FileNotFoundError(msg)

                ds_type = stream_info["type"]
                if is_root():
                    logger.info(
                        f"Opening dataset with type: {ds_type}"
                        + f" from stream config {stream_info['name']}.",
                    )
                ds = dataset(filename=filename, **kwargs)

                fsm = self.forecast_steps[0]
                if len(ds) > 0:
                    self.len = min(self.len, len(ds) - (self.len_hrs * (fsm + 1)) // self.step_hrs)

                # MODIFIES config !!!
                stream_info[str(self._stage) + "_source_channels"] = ds.source_channels
                stream_info[str(self._stage) + "_target_channels"] = ds.target_channels
                stream_info["target_channel_weights"] = (
                    ds.target_channel_weights
                    if ds.target_channel_weights is not None
                    else [1.0 for _ in ds.target_channels]
                )

                self.streams_datasets[-1] += [ds]

        index_range = self.time_window_handler.get_index_range()
        self.len = int(index_range.end - index_range.start)
        self.len = min(self.len, samples_per_epoch if samples_per_epoch else self.len)
        # adjust len to split loading across all workers and ensure it is multiple of batch_size
        len_chunk = ((self.len // cf.world_size) // batch_size) * batch_size
        self.len = min(self.len, len_chunk)
        logger.info(f"index_range={index_range}, len={self.len}, len_chunk={len_chunk}")

        self.rank = cf.rank
        self.world_size = cf.world_size

        self.streams = cf.streams
        self.shuffle = shuffle
        self.input_window_steps = cf.input_window_steps
        self.embed_local_coords = cf.embed_local_coords
        self.embed_centroids_local_coords = cf.embed_centroids_local_coords
        self.sampling_rate_target = cf.sampling_rate_target

        self.batch_size = batch_size

        # ensure data_loader_rng_seed is not smaller than loader_num_workers to avoid
        # issues in per loader rng seed computation
        self.data_loader_rng_seed = (
            cf.data_loader_rng_seed
            if cf.data_loader_rng_seed > cf.loader_num_workers
            else cf.data_loader_rng_seed * 13
        )

        self.healpix_level: int = cf.healpix_level
        self.num_healpix_cells: int = 12 * 4**self.healpix_level

        # Initialize tokenizer based on training mode
        self.training_mode = cf.get("training_mode", "forecast")
        
        if self.training_mode == "forecast":
            self.tokenizer = TokenizerForecast(cf.healpix_level)
            self.use_student_teacher = False
            
        elif self.training_mode == "masking":
            masker = Masker(cf)
            self.tokenizer = TokenizerMasking(cf.healpix_level, masker)
            self.use_student_teacher = False
            assert self.forecast_offset == 0, "masked token modeling requires auto-encoder training"
            assert self.input_window_steps == 1, (
                "masked token modeling does not support input_window_steps > 1"
            )
            
        elif self.training_mode == "student_teacher":
            masker = Masker(cf)
            self.tokenizer = TokenizerMasking(cf.healpix_level, masker)
            self.use_student_teacher = True
            assert self.forecast_offset == 0, "student-teacher training requires auto-encoder mode"
            # TODO
            assert self.input_window_steps == 1, (
                "student-teacher does not support input_window_steps > 1"
            )
            logger.info(
                f"Student-teacher mode enabled: "
                f"teacher strategy={cf.student_teacher['global']['strategy']}, "
                f"num_students={cf.student_teacher['locals']['num_views']}"
            )
            
        else:
            assert False, f"Unsupported training mode: {cf.training_mode}"

        self.epoch = 0

    def advance(self):
        """Advance epoch (this is applied to the template for the worker processes)"""
        self.epoch += 1

    ###################################################
    def get_sources_size(self):
        return [
            ds[0].get_source_num_channels()
            + ds[0].get_geoinfo_size()
            + ds[0].get_coords_size()
            + self.tokenizer.get_size_time_embedding()
            for ds in self.streams_datasets
        ]

    ###################################################
    def get_sources_num_channels(self):
        return [ds[0].get_source_num_channels() for ds in self.streams_datasets]

    ###################################################
    def get_targets_num_channels(self):
        return [ds[0].get_target_num_channels() for ds in self.streams_datasets]

    ###################################################
    def get_targets_coords_size(self):
        # TODO: avoid hard coding magic values
        # +6 at the end for stream_id and time encoding
        return [
            (ds[0].get_geoinfo_size() + (5 * (3 * 5)) + 3 * 8) + 6 for ds in self.streams_datasets
        ]

    ###################################################
    def reset(self):
        # initialize the random number generator: self.data_loader_rng_seed is set to a DDP-unique
        # value in worker_workset()
        self.rng = np.random.default_rng(self.data_loader_rng_seed)

        fsm = (
            self.forecast_steps[min(self.epoch, len(self.forecast_steps) - 1)]
            if self.forecast_policy != "random"
            else self.forecast_steps.max()
        )
        if fsm > 0:
            logger.info(f"forecast_steps at epoch={self.epoch} : {fsm}")

        # data
        index_range = self.time_window_handler.get_index_range()
        idx_end = index_range.end
        # native length of datasets, independent of epoch length that has potentially been specified
        forecast_len = (self.len_hrs * (fsm + 1)) // self.step_hrs
        idx_end -= forecast_len + self.forecast_offset
        assert idx_end > 0, "dataset size too small for forecast range"
        self.perms = np.arange(index_range.start, idx_end)
        if self.shuffle:
            self.perms = self.rng.permutation(self.perms)

        # forecast time steps
        len_dt_samples = len(self) // self.batch_size
        if self.forecast_policy is None:
            self.perms_forecast_dt = np.zeros(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "fixed" or self.forecast_policy == "sequential":
            self.perms_forecast_dt = fsm * np.ones(len_dt_samples, dtype=np.int64)
        elif self.forecast_policy == "random" or self.forecast_policy == "sequential_random":
            self.perms_forecast_dt = self.rng.integers(
                low=self.forecast_steps.min(), high=fsm + 1, size=len_dt_samples, dtype=np.int64
            )
        else:
            assert False

        self.tokenizer.reset_rng(self.rng)

    def denormalize_source_channels(self, stream_id, data) -> torch.Tensor:
        return self.streams_datasets[stream_id][0].denormalize_source_channels(data)

    def denormalize_target_channels(self, stream_id, data) -> torch.Tensor:
        return self.streams_datasets[stream_id][0].denormalize_target_channels(data)

    def _build_stream_data_for_view(
        self,
        idx: TIndex,
        forecast_dt: int,
        view_meta: ViewMetadata,
        stream_info: dict,
        stream_ds: list,
    ) -> StreamData:
        """
        Build a StreamData object for a single view (teacher or student).
        
        Args:
            idx: Time index for this sample
            forecast_dt: Number of forecast steps
            view_meta: ViewMetadata describing spatial mask
            stream_info: Stream configuration dict
            stream_ds: List of dataset readers for this stream
            
        Returns:
            StreamData with source and targets masked according to view_meta
        """
        time_win_source = self.time_window_handler.window(idx)
        
        stream_data = StreamData(idx, forecast_dt + self.forecast_offset, self.num_healpix_cells)
        
        # Collect and tokenize source data with view mask applied
        rdata_src = collect_datasources(stream_ds, idx, "source")
        
        if rdata_src.is_empty():
            rdata_src = spoof(
                self.healpix_level,
                time_win_source.start,
                stream_ds[0].get_geoinfo_size(),
                stream_ds[0].mean[stream_ds[0].source_idx],
            )
            stream_data.source_is_spoof = True
        
        # Apply view mask during tokenization
        with self.tokenizer.use_keep_cells(view_meta.keep_mask):
            (ss_cells, ss_lens, ss_centroids) = self.tokenizer.batchify_source(
                stream_info,
                readerdata_to_torch(rdata_src),
                (time_win_source.start, time_win_source.end),
                stream_ds[0].normalize_coords,
            )
        
        stream_data.add_source(rdata_src, ss_lens, ss_cells, ss_centroids)
        
        # Collect and tokenize targets for all forecast steps
        for fstep in range(self.forecast_offset, self.forecast_offset + forecast_dt + 1):
            step_forecast_dt = idx + (self.forecast_delta_hrs * fstep) // self.step_hrs
            time_win_target = self.time_window_handler.window(step_forecast_dt)
            
            rdata_tgt = collect_datasources(stream_ds, step_forecast_dt, "target")
            
            if rdata_tgt.is_empty():
                rdata_tgt = spoof(
                    self.healpix_level,
                    time_win_target.start,
                    stream_ds[0].get_geoinfo_size(),
                    stream_ds[0].mean[stream_ds[0].target_idx],
                )
                stream_data.target_is_spoof = True
            
            # Target uses the same view mask (set by mask_source via perm_sel)
            (tt_cells, tc, tt_c, tt_t) = self.tokenizer.batchify_target(
                stream_info,
                self.sampling_rate_target,
                readerdata_to_torch(rdata_tgt),
                (time_win_target.start, time_win_target.end),
            )
            
            stream_data.add_target(fstep, tt_cells, tc, tt_c, tt_t)
        
        return stream_data

    def __iter__(self):
        """
        Yield batches of data.
        
        Returns:
            For forecast/masking modes:
                (batch, source_cell_lens, target_coords_idx, forecast_dt)
            
            For student_teacher mode:
                (model_batch, source_cell_lens, target_coords_idx, forecast_dt)
                where model_batch is a ModelBatch containing teacher + students + metadata
        """
        iter_start, iter_end = self.worker_workset()
        logger.info(f"iter_start={iter_start}, iter_end={iter_end}, len={self.len}")

        self.reset()

        idx_raw = iter_start
        for i, _bidx in enumerate(range(iter_start, iter_end, self.batch_size)):
            forecast_dt = self.perms_forecast_dt[i]

            batch = []  # list[list[StreamData]] for normal modes
            model_batches = [] if self.use_student_teacher else None  # list[ModelBatch] for ST mode
            
            while len(batch) < self.batch_size:
                idx: TIndex = self.perms[idx_raw % self.perms.shape[0]]
                idx_raw += 1

                time_win_source = self.time_window_handler.window(idx)

                # Sample masking strategy once per batch item (if using combination mode)
                if hasattr(self.tokenizer, "masker"):
                    self.tokenizer.masker.set_batch_strategy()

                if self.use_student_teacher:
                    # Generate teacher and student views for all streams
                    teacher_streams = []  # list[StreamData], one per stream
                    student_streams_all = []  # list[list[StreamData]], [n_students][n_streams]
                    teacher_view_meta = None
                    student_view_metas = []

                    for stream_info, stream_ds in zip(self.streams, self.streams_datasets, strict=True):
                        # Collect source data (coords/geoinfos/data) for this sample
                        rdata_src = collect_datasources(stream_ds, idx, "source")

                        if rdata_src.is_empty():
                            rdata_src = spoof(
                                self.healpix_level,
                                time_win_source.start,
                                stream_ds[0].get_geoinfo_size(),
                                stream_ds[0].mean[stream_ds[0].source_idx],
                            )

                        rdata_torch = readerdata_to_torch(rdata_src)

                        # Generate views once per sample (shared across streams), no tokenization here
                        if teacher_view_meta is None:
                            teacher_view_meta, student_view_metas = self.tokenizer.make_views_for_sample(
                                rdata_torch
                            )

                        # Build teacher StreamData for this stream with mask applied
                        teacher_sd = self._build_stream_data_for_view(
                            idx, forecast_dt, teacher_view_meta, stream_info, stream_ds
                        )
                        teacher_streams.append(teacher_sd)

                        # Build student StreamData for each local view
                        if len(student_streams_all) == 0:
                            student_streams_all = [[] for _ in student_view_metas]

                        for view_idx, student_meta in enumerate(student_view_metas):
                            student_sd = self._build_stream_data_for_view(
                                idx, forecast_dt, student_meta, stream_info, stream_ds
                            )
                            student_streams_all[view_idx].append(student_sd)

                    # Reset strategy for next batch item
                    if hasattr(self.tokenizer, "masker"):
                        self.tokenizer.masker.reset_batch_strategy()

                    # Skip if all targets empty
                    if all(s.empty() or s.target_empty() for s in teacher_streams):
                        continue

                    # Package ModelBatch (students + teacher + metadata)
                    model_batch = ModelBatch(
                        model_inputs=student_streams_all,
                        targets=[teacher_streams],
                        view_metadata=[teacher_view_meta] + student_view_metas,
                        batch_info={"sample_idx": int(idx), "forecast_dt": int(forecast_dt)},
                    )
                    
                    # **********************************************************************
                    # NOTE:
                    # So note at this point we have:
                    # ModelBatch containing a lot of model_inputs (which are StreamData objects of local views)
                    # Contains one target, which is a single Teacher StreamData object
                    # View metadata describing the masks used, giving a view_id, and some information on the mask as the strategy and rate
                    # For the students, this also has information on the parent global view that corresponds to this view.
                    # NOTE: I am not sure how best it is to access this. NOTE, later on we do not pass this information on at the end of MultiStreamDataSampler
                    # BUT we will need to, including for the diffusion random number
                    # So we need to think about how best to pass this on.
                    # There is also batch_info.
                    # There is also the functionality to include the source_cell_lens and the target_coords_idx for the student views here too.
                    # This is weird at the moment as we say we can do it for student but not the teacher.

                    # import pdb; pdb.set_trace()
                    
                    model_batches.append(model_batch)
                    batch.append(teacher_streams)  # for compatibility with compute_source_cell_lens #TODO
                    
                else:
                    # ===== FORECAST / MASKING MODE =====
                    # This is aiming to proceed more or less as before...
                    streams_data: list[StreamData] = []
                    
                    for stream_info, stream_ds in zip(self.streams, self.streams_datasets, strict=True):
                        stream_data = StreamData(
                            idx, forecast_dt + self.forecast_offset, self.num_healpix_cells
                        )
                        
                        rdata_src = collect_datasources(stream_ds, idx, "source")
                        
                        if rdata_src.is_empty():
                            rdata_src = spoof(
                                self.healpix_level,
                                time_win_source.start,
                                stream_ds[0].get_geoinfo_size(),
                                stream_ds[0].mean[stream_ds[0].source_idx],
                            )
                            stream_data.source_is_spoof = True
                        
                        (ss_cells, ss_lens, ss_centroids) = self.tokenizer.batchify_source(
                            stream_info,
                            readerdata_to_torch(rdata_src),
                            (time_win_source.start, time_win_source.end),
                            stream_ds[0].normalize_coords,
                        )
                        
                        stream_data.add_source(rdata_src, ss_lens, ss_cells, ss_centroids)
                        
                        for fstep in range(
                            self.forecast_offset, self.forecast_offset + forecast_dt + 1
                        ):
                            step_forecast_dt = idx + (self.forecast_delta_hrs * fstep) // self.step_hrs
                            time_win_target = self.time_window_handler.window(step_forecast_dt)
                            
                            rdata_tgt = collect_datasources(stream_ds, step_forecast_dt, "target")
                            
                            if rdata_tgt.is_empty():
                                rdata_tgt = spoof(
                                    self.healpix_level,
                                    time_win_target.start,
                                    stream_ds[0].get_geoinfo_size(),
                                    stream_ds[0].mean[stream_ds[0].target_idx],
                                )
                                stream_data.target_is_spoof = True
                            
                            (tt_cells, tc, tt_c, tt_t) = self.tokenizer.batchify_target(
                                stream_info,
                                self.sampling_rate_target,
                                readerdata_to_torch(rdata_tgt),
                                (time_win_target.start, time_win_target.end),
                            )
                            
                            stream_data.add_target(fstep, tt_cells, tc, tt_c, tt_t)
                        
                        streams_data.append(stream_data)
                    
                    # Reset strategy for next batch item
                    if hasattr(self.tokenizer, "masker"):
                        self.tokenizer.masker.reset_batch_strategy()
                    
                    if not (all(s.empty() or s.target_empty() for s in streams_data)):
                        batch.append(streams_data)

            # Compute offsets and indices for teacher batch...
            # TODO: we had the weird thing above batch.append(teacher_streams) so we are compatible with this 
            source_cell_lens = compute_source_cell_lens(batch)
            batch = compute_offsets_scatter_embed(batch)
            target_coords_idx = compute_idxs_predict(self.forecast_offset + forecast_dt, batch)

            # ...
            assert len(batch) == self.batch_size
            
            
            # NOTE: THIS IS THE PREPARATION OF THE STUDENT VIEWS
            # THESE ARE NOT CURRENTLY USED, and is very rough
            # this has not been stress-tested for delivery to the model
            # but this is now preparing the student views for delivery to
            # the model as it is written now 13/11/25
            if self.use_student_teacher:
                for mb in model_batches:
                    # Initialize storage lists
                    mb.student_source_cell_lens = []
                    mb.student_target_coords_idx = []
                    
                    # Compute for each student view
                    for student_view_streams in mb.model_inputs:
                        # Wrap in list to match expected shape for compute functions
                        student_batch = [student_view_streams]
                        
                        # Compute and store source cell lengths
                        student_source_lens = compute_source_cell_lens(student_batch)
                        mb.student_source_cell_lens.append(student_source_lens)
                        
                        # Compute offsets (modifies StreamData in-place...)
                        compute_offsets_scatter_embed(student_batch)
                        
                        # Compute and store target coordinate indices
                        student_target_idx = compute_idxs_predict(
                            self.forecast_offset + forecast_dt, 
                            student_batch
                        )
                        mb.student_target_coords_idx.append(student_target_idx)
                
                # NOTE: for current simplicity, we are returning source_cell_lens and target_coords_idx as before
                # NOTE: here these correspond to the teacher view only
                # TODO: change this. Wrap in ModelBatch too?
                # TODO: wrap everything in ModelBatch. To do tomorrow morning. This is bizarre.
                yield (model_batches, source_cell_lens, target_coords_idx, forecast_dt)
                #                       ^^^^^^^^^ teacher ones ^^^^^^     
            else:                      
                yield (batch, source_cell_lens, target_coords_idx, forecast_dt)

    def __len__(self):
        return self.len

    def worker_workset(self):
        """Determine work range for this worker/rank"""
        local_start, local_end = self.rank * self.len, (self.rank + 1) * self.len

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            assert self.world_size == 1, self.world_size
            iter_start = 0
            iter_end = len(self)
        else:
            dist = torch.distributed
            self.data_loader_rng_seed *= (
                (((dist.get_rank() + 1) * 73) if dist.is_initialized() else 1)
                * ((worker_info.id + 1) * 37)
                * (self.epoch + 13)
                * 7
            )
            per_worker = (local_end - local_start) // worker_info.num_workers
            iter_start = local_start + worker_info.id * per_worker
            iter_end = iter_start + per_worker
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = local_end
            logger.info(
                f"{self.rank}::{worker_info.id}"
                + f" : dataset [{local_start},{local_end}) : [{iter_start},{iter_end})"
            )

        return iter_start, iter_end