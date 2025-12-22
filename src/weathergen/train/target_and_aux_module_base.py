import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class TargetAuxOutput:
    """
    A dataclass to encapsulate the TargetAndAuxCalculator output and give a clear API.
    """

    num_forecast_steps: int

    physical: dict[str, torch.Tensor]
    latent: dict[str, torch.Tensor]
    aux_outputs: dict[str, torch.Tensor]


class TargetAndAuxModuleBase:
    def __init__(self, cf, model, **kwargs):
        pass

    def reset(self):
        pass

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass

    def compute(self, sample, *args, **kwargs) -> TargetAuxOutput:
        pass

    def to_device(self, device):
        pass


class PhysicalTargetAndAux(TargetAndAuxModuleBase):
    def __init__(self, cf, model, **kwargs):
        return

    def reset(self):
        return

    def update_state_pre_backward(self, istep, batch, model, **kwargs):
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs):
        return

    def compute(self, batch, *args, **kwargs) -> TargetAuxOutput:
        # TODO: properly retrieve/define these
        stream_names = [k for k, _ in batch.target_samples[0].streams_data.items()]
        forecast_steps = batch.get_num_target_steps()

        # collect all targets, concatenating across batch dimension since this is also how it
        # happens for predictions in the model
        targets, aux_outputs = {}, {}
        for stream_name in stream_names:
            # collect targets for all forecast steps
            targets[stream_name] = []
            for fstep in range(forecast_steps):
                targets_cur, target_times_cur, target_coords_cur = [], [], []
                for sample in batch.target_samples:
                    targets_cur += [sample.streams_data[stream_name].target_tokens[fstep]]
                    target_times_cur += [sample.streams_data[stream_name].target_times_raw[fstep]]
                    target_coords_cur += [sample.streams_data[stream_name].target_coords_raw[fstep]]

                targets[stream_name].append(
                    {
                        "target": torch.cat(targets_cur),
                        "target_times": np.concatenate(target_times_cur),
                        "target_coords": np.concatenate(target_coords_cur),
                    }
                )

            # use aux_outputs to collect spoof flag
            aux_outputs[stream_name] = {"is_spoof": sample.streams_data[stream_name].is_spoof()}

        return TargetAuxOutput(forecast_steps, targets, None, aux_outputs)

    def to_device(self, device):
        return
