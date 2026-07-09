# train/ — training loop

- `trainer.py` / `trainer_base.py` — `Trainer`: distributed setup, step loop, checkpointing.
- `loss_calculator.py` — `LossCalculator`, weighted sum over loss terms; the loss classes themselves (physical, SSL) live in `loss_modules/`, instantiated by config `type`. `lr_scheduler.py` — `LearningRateScheduler`.
- `target_and_aux_*.py`, `teacher_utils.py` — target/aux computation for SSL (`EncoderTeacher` for JEPA-style training), selected via `get_target_aux_calculator`.
- `collapse_monitor.py` — `CollapseMonitor`, detects representation collapse in SSL runs.
- Step anatomy + invariants: `agent_docs/training-step.md`; teacher/latent-loss machinery: `agent_docs/ssl-training.md`. Read before changing the loop or losses; update after.
