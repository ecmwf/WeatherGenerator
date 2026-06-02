from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.utils import validation_io


class FakeModelOutput:
    def __init__(self, preds_by_stream):
        self.preds_by_stream = preds_by_stream

    def get_physical_prediction(self, fstep, stream_name):
        return self.preds_by_stream.get((fstep, stream_name))


class FakeSourceSample:
    def __init__(self, sample_idx):
        self.streams_data = {
            "source": SimpleNamespace(source_raw=[np.zeros((1, 1), dtype=np.float32)], sample_idx=sample_idx)
        }


class FakeSourceSamples:
    def __init__(self, sample_idxs):
        self.samples = [FakeSourceSample(sample_idx) for sample_idx in sample_idxs]

    def get_samples(self):
        return self.samples


class FakeBatch:
    def __init__(self, output_idxs, sample_idxs):
        self.output_idxs = output_idxs
        self.source_samples = FakeSourceSamples(sample_idxs)

    def get_output_idxs(self):
        return self.output_idxs

    def get_source_samples(self):
        return self.source_samples


def test_write_output_skips_spoofed_forcing_stream_without_predictions():
    cf = SimpleNamespace(streams=[{"name": "OBS", "pred_head": {"ens_size": 1}}])
    val_cfg = OmegaConf.create({"losses": {"physical": {"type": "LossPhysical"}}})
    batch = FakeBatch([0], [0])
    model_output = FakeModelOutput({})
    target_aux_out = {
        "physical": SimpleNamespace(
            physical=[
                {
                    "OBS": {
                        "target": [torch.ones((1, 1))],
                        "target_coords": [torch.ones((1, 2))],
                        "target_times": [np.array(["2023-10-01T00:00:00"], dtype="datetime64[ns]")],
                        "is_spoof": [True],
                        "idxs_inv": [None],
                    }
                }
            ]
        )
    }

    validation_io.write_output(
        cf,
        val_cfg,
        1,
        0,
        0,
        lambda _stream, tensor: tensor,
        batch,
        model_output,
        target_aux_out,
    )


def test_write_output_preserves_non_spoof_samples(monkeypatch):
    captured = {}

    class FakeOutputBatchData:
        def __init__(self, *args):
            captured["targets_all"] = args[2]
            captured["preds_all"] = args[3]

        def items(self):
            return [self]

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_zarr(self, subset):
            captured["written_subset"] = subset

    class FakeTimeWindowHandler:
        def __init__(self, *args):
            return

        def window(self, idx):
            return SimpleNamespace(
                start=np.datetime64("2023-10-01T00:00:00"),
                end=np.datetime64("2023-10-01T12:00:00"),
            )

    monkeypatch.setattr(validation_io.io, "OutputBatchData", FakeOutputBatchData)
    monkeypatch.setattr(validation_io, "zarrio_writer", lambda _path: FakeWriter())
    monkeypatch.setattr(validation_io.config, "get_path_results", lambda _cf, _mini_epoch: Path("/tmp/fake.zarr"))
    monkeypatch.setattr(validation_io, "TimeWindowHandler", FakeTimeWindowHandler)

    cf = OmegaConf.create(
        {
            "streams": [
                {
                    "name": "ERA5",
                    "pred_head": {"ens_size": 1},
                    "val_target_channels": ["t2m"],
                    "val_source_channels": ["t2m"],
                }
            ],
            "rank": 0,
            "zarr_store": "zip",
        }
    )
    val_cfg = OmegaConf.create(
        {
            "losses": {"physical": {"type": "LossPhysical"}},
            "output": {},
            "start_date": "2023-10-01T00:00",
            "end_date": "2023-10-02T00:00",
            "time_window_len": "12:00:00",
            "time_window_step": "12:00:00",
        }
    )
    batch = FakeBatch([0], [0, 1])
    model_output = FakeModelOutput(
        {
            (0, "ERA5"): [
                torch.ones((1, 1, 1), dtype=torch.float32),
                2 * torch.ones((1, 1, 1), dtype=torch.float32),
            ]
        }
    )
    target_aux_out = {
        "physical": SimpleNamespace(
            physical=[
                {
                    "ERA5": {
                        "target": [
                            torch.ones((1, 1), dtype=torch.float32),
                            2 * torch.ones((1, 1), dtype=torch.float32),
                        ],
                        "target_coords": [
                            torch.ones((1, 2), dtype=torch.float32),
                            2 * torch.ones((1, 2), dtype=torch.float32),
                        ],
                        "target_times": [
                            np.array(["2023-10-01T00:00:00"], dtype="datetime64[ns]"),
                            np.array(["2023-10-01T12:00:00"], dtype="datetime64[ns]"),
                        ],
                        "is_spoof": [True, False],
                        "idxs_inv": [None, None],
                    }
                }
            ]
        )
    }

    validation_io.write_output(
        cf,
        val_cfg,
        2,
        0,
        0,
        lambda _stream, tensor: tensor,
        batch,
        model_output,
        target_aux_out,
    )

    assert captured["preds_all"][0][0].shape == (1, 1, 1)
    assert captured["targets_all"][0][0].shape == (1, 1)