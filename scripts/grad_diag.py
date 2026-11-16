"""Per-lead-window gradient diagnostics for pushforward CRPS training (no training, no weight updates).

The real trainer builds the model, data and loss exactly as in training; GradDiagTrainer replaces the training
loop. For each training sample (one init date, its ensemble members) and each window w it computes

    g_w = d L_w / d theta        (theta: all trainable parameters = the forecast engine)

where L_w is the training loss (fair CRPS, channel and lead-balance weights, mean over the window's steps)
restricted to window w, with gradient only through the window's own steps, exactly as pushforward training does:
  - anchor: steps 1-4 (scored with gradient in every training batch)
  - final K: steps K-3..K (scored with gradient when the batch's rollout length is K; earlier steps run without
    gradient). One training batch with rollout length K has gradient (g_anchor + g_final(K)) / 2.
Each window is a separate forward/backward pass with its own rollout plan (batch.rollout_plan); the model code is
unchanged. Dropout is active as in training, so each g_w also contains dropout noise.

Saved per process (to OUT_DIR): S[w] = sum over samples of g_w (float32, for the cross-sample terms), and per
parameter tensor the same-sample Gram sum_i <g_w,i, g_v,i>, plus losses. scripts/grad_diag_summary.py combines
processes: the inner product of the MEAN gradients is estimated without each sample's own noise as
    <mu_w, mu_v> ~ ( <S_w, S_v> - sum_i <g_w,i, g_v,i> ) / ( N (N-1) ).

Usage (one process per GPU, world size 1 each, from the WeatherGenerator root):
  WORLD_SIZE=1 python scripts/grad_diag.py --out-dir DIR --proc P --num-samples N [--windows anchor,8,12,...]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from weathergen.common import config
from weathergen.common.logger import init_loggers
from weathergen.model.utils import apply_fct_to_blocks, set_to_eval
from weathergen.train.trainer import Trainer
from weathergen.train.utils import extract_batch_metadata, get_target_idxs_from_cfg
from weathergen.utils.rollout import RolloutPlan

BASE = "./config/config_nhv6tkln_chkpt8_base.yml"
CONFIGS = [
    "./config/config_gjl_repro_finetune_forecast_n320_rollout_8_nodes.yml",
    "./config/config_long_rollout_pushforward.yml",
    "./config/config_long_rollout_anchor.yml",
    "./config/config_long_rollout_crps.yml",
    "./config/config_lead_balance_c8lx80.yml",
]


class GradDiagTrainer(Trainer):
    def validate_before_training(self):
        pass

    def train(self, mini_epoch):
        a = self.diag
        cf = self.cf
        self.model.train()
        apply_fct_to_blocks(self.model, cf.freeze_modules, set_to_eval)
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        names = [n for n, _ in params]
        sizes = [p.numel() for _, p in params]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        P = int(offs[-1])
        W = len(a.windows)
        print(f"[grad_diag] trainable tensors {len(names)}, parameters {P / 1e6:.1f} M, windows {a.windows}", flush=True)
        S = torch.zeros(W, P, dtype=torch.float32)
        gram_same = np.zeros((len(names), W, W))
        losses = np.full((a.num_samples, W), np.nan)
        g = torch.zeros(W, P, dtype=torch.float32)
        it = iter(self.data_loader)
        t0 = time.time()
        for i in range(a.num_samples):
            batch = next(it)
            batch.to_device(self.device)
            steps = list(batch.get_output_idxs())
            assert len(steps) == 80, f"expected 80 output steps, got {len(steps)}"
            assert getattr(batch, "ensemble_size", 1) == 2, getattr(batch, "ensemble_size", None)
            for w, win in enumerate(a.windows):
                loss_steps = steps[:4] if win == "anchor" else steps[int(win) - 4 : int(win)]
                batch.rollout_plan = RolloutPlan(
                    no_grad_steps=frozenset(steps) - frozenset(loss_steps), loss_steps=frozenset(loss_steps)
                )
                for n, p in params:
                    p.grad = None
                with torch.autocast(device_type=f"cuda:{cf.local_rank}", dtype=self.mixed_precision_dtype,
                                    enabled=cf.with_mixed_precision):
                    preds = self.model(self.model_params, batch.get_source_samples())
                    targets_and_auxs = {}
                    for loss_name, target_aux in self.target_and_aux_calculators.items():
                        target_idxs = get_target_idxs_from_cfg(self.training_cfg, loss_name)
                        targets_and_auxs[loss_name] = target_aux.compute(
                            cf.general.istep, batch.get_target_samples(target_idxs), self.model_params, self.model
                        )
                loss = self.loss_calculator.compute_loss(
                    preds=preds, targets_and_aux=targets_and_auxs, metadata=extract_batch_metadata(batch),
                    istep=cf.general.istep,
                )
                loss.backward()
                losses[i, w] = float(loss.detach())
                for k, (n, p) in enumerate(params):
                    gk = p.grad if p.grad is not None else torch.zeros_like(p)
                    g[w, offs[k] : offs[k + 1]] = gk.detach().float().flatten().cpu()
            S += g
            for k in range(len(names)):
                gk = g[:, offs[k] : offs[k + 1]].double()
                gram_same[k] += (gk @ gk.T).numpy()
            tot = g.double().norm(dim=1).numpy()
            print(f"[grad_diag] sample {i + 1}/{a.num_samples} {time.time() - t0:.0f}s losses "
                  + " ".join(f"{x:.4g}" for x in losses[i]) + " | |g| " + " ".join(f"{x:.3g}" for x in tot), flush=True)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(S, out / f"S_proc{a.proc}.pt")
        np.savez(out / f"stats_proc{a.proc}.npz", gram_same=gram_same, losses=losses, offs=offs,
                 windows=np.array(a.windows), names=np.array(names), num_samples=a.num_samples)
        json.dump({"names": names, "sizes": sizes, "windows": a.windows, "num_samples": a.num_samples,
                   "rng_seed": int(cf.data_loading.rng_seed), "elapsed_s": time.time() - t0},
                  open(out / f"meta_proc{a.proc}.json", "w"))
        print(f"[grad_diag] done in {time.time() - t0:.0f}s, wrote {out}", flush=True)
        raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--proc", type=int, required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--windows", default="anchor,8,12,16,20,28,40,60,80")
    ap.add_argument("--load-run-id", default="axhby5w8")
    ap.add_argument("--load-mini-epoch", type=int, default=9)
    ap.add_argument("--private-config", required=True)
    a = ap.parse_args()
    a.windows = [w if w == "anchor" else str(int(w)) for w in a.windows.split(",")]
    assert os.environ.get("WORLD_SIZE") == "1", "run one process per GPU with WORLD_SIZE=1"
    options = [
        f"load_chkpt.run_id={a.load_run_id}", f"load_chkpt.mini_epoch={a.load_mini_epoch}",
        "streams_directory=./config/streams/gjl6a0kb_reproduction_n320_new_channels_forecast_iasi_channels_mnt_cw/",
        "fe_impute_latent_noise_std=1.0e-05",
        "training_config.forecast.lead_balance.power=0.5",
        "training_config.forecast.rollout_anchor_steps=4",
        "training_config.forecast.rollout_grad_steps=4",
        "training_config.forecast.rollout_loss_steps=4",
        # every batch carries targets for 80 steps; the window's rollout plan selects the scored steps
        "training_config.forecast.policy=fixed",
        "training_config.forecast.num_steps=80",
        "with_ddp=false",
        "with_fsdp=false",
        "data_loading.num_workers=4",
    ]
    cf = config.load_merge_configs(a.private_config, None, None, BASE, *CONFIGS, config.from_cli_arglist(options))
    cf = config.set_run_id(cf, f"gdg{a.proc:01d}{a.load_run_id[:4]}", False)
    cf.data_loading.rng_seed = 20261116 + 1000 * a.proc  # distinct, reproducible training dates per process
    cf.stage = "train"
    devices = Trainer.init_torch(multiprocessing_method=cf.general.get("multiprocessing_method", "fork"))
    cf = Trainer.init_ddp(cf)
    init_loggers(cf.general.run_id)
    cf.streams = config.load_streams(Path(cf.streams_directory))
    trainer = GradDiagTrainer(cf.train_logging)
    trainer.diag = a
    trainer.run(cf, devices)
    print("[grad_diag] training loop was not reached", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
