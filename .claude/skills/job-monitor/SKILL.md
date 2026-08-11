---
name: job-monitor
description: Hourly SLURM job monitor for WeatherGenerator pipeline runs on santis. Discovers runs, triages crashes, cancels dependent jobs, runs an investigate→verify→fix→verify agent pipeline, resubmits (max 2 auto-resubmits per run, then approval-gated), and emails a summary per incident. Start the loop session with `claude --settings .claude/skills/job-monitor/settings.json` (sets Sonnet + the minimal permission allowlist), then run `/loop 1h /job-monitor`; incident agents are pinned to Opus.
---

# Job monitor — one pass

You are the MONITOR. You are the **only** entity allowed to call the launcher
(`launch-slurm-sophie.py`), `sbatch`, or `scancel`. Subagents you spawn must never
submit or cancel jobs. Do exactly one pass, then end your turn (the /loop handles cadence).

## Constants

```
WGEN            = /users/sxhonneu/projects/sophie-dev/WeatherGenerator
WGEN_PRIVATE    = /users/sxhonneu/projects/sophie-dev/WeatherGenerator-private
LAUNCHER        = $WGEN_PRIVATE/hpc/launch-slurm-sophie.py     (run with cwd=$WGEN_PRIVATE)
SLURM_ROOT      = /iopsstor/scratch/cscs/thunter/slurm
RUN_DIR(id)     = $SLURM_ROOT/slurm_weathergen_<id>_dir
RUN_LOGS(id)    = RUN_DIR(id)/WeatherGenerator/logs/            (weathergen-<jobname>.<jobid>.{out,err})
RUN_OUTPUT(id)  = RUN_DIR(id)/WeatherGenerator/output/          (output_<stage_run_id>_<jobid>.txt)
JOBS_CSV        = ~/.weathergen/jobs.csv                        (written by the launcher on every live launch)
STATE           = $WGEN/notes/job-monitor/state.yml
INCIDENTS       = $WGEN/notes/job-monitor/incidents/
APPROVALS       = $WGEN/notes/job-monitor/approvals/
EMAIL           = lpxhonneux@gmail.com
CONFIGS_REPO    = /users/sxhonneu/projects/sophie-dev/weathergen-configs   (a separate git repo;
                   many runs' --config-dir points here — check JOBS_CSV before assuming a
                   stage's config lives under $WGEN/config/)
```

Job names: `weathergen_<run_id>-<stage>_part<N>` (train/inference chains),
`weathergen_<run_id>_cleanup`, `weathergen_<run_id>_inference`. Stage run ids are
`<run_id>-<stage>`; the *pipeline* run id is the part before the first `-`.

**Where the logs actually are** (three places, in order of usefulness for a crash):
1. `RUN_OUTPUT/output_<stage_run_id>_<jobid>.txt` — python's real stdout+stderr
   (the sbatch script redirects srun with `&>` here; inference jobs:
   `output_inference_<run_id>_<jobid>.txt`). **Tracebacks land here, nowhere else.**
2. `RUN_LOGS/<stage_run_id>/log.txt` and `error.txt` — the application logger's output.
3. `RUN_LOGS/weathergen-<jobname>.<jobid>.{out,err}` — SLURM wrapper only
   (env setup + "Finished job"); `.err` is normally empty. Near-useless for triage.
Note `RUN_DIR/WeatherGenerator/logs` is a symlink to a shared logs dir.

## Pass algorithm

### 0. Load state
Read `STATE` (create skeleton `{last_check: <now - 24h>, runs: {}}` if missing).
Check `APPROVALS/` for approval files (see step 5). Also honor a user message in the
session like "approve <run_id>" as an approval.

### 1. Discover terminal jobs
```
sacct -u sxhonneu -S <last_check minus 2h> -X -n -P -o "JobID,JobName,State,ExitCode,End,Elapsed,Timelimit"
```
**Permission hygiene (applies to every Bash command in this skill):** no shell
variables, no assignments (`RUN_DIR=...; ls $RUN_DIR`), no command substitution
(`$(date ...)`, `$(whoami)`), no `$USER`/`~` where a literal path works — any of
these bypasses the permission allowlist and forces a manual prompt. Always spell
out literal absolute paths and the literal username `sxhonneu`.
Keep rows where JobName matches `^weathergen_` and State is terminal
(COMPLETED, FAILED, TIMEOUT, NODE_FAIL, OUT_OF_MEMORY, PREEMPTED, CANCELLED, BOOT_FAIL).
Skip JobIDs already in `runs.<id>.handled_job_ids`. Parse run_id and stage from JobName.

### 1b. Verify COMPLETED jobs at step level — job state lies
The sbatch wrapper does **not** propagate srun's exit code: a job whose python
step crashed still ends `COMPLETED` (the j0qtd6ip pipeline "completed" 6 stages
in 1–3 min each this way). For every new COMPLETED train/inference job run:
```
sacct -j <id1>,<id2>,... -n -P -o "JobID,JobName,State,ExitCode,DerivedExitCode,Elapsed"
```
(step level — no `-X`) and check the `python` step (`<jobid>.0`):
- python step FAILED (or DerivedExitCode != 0:0) → reclassify as **CRASH** (step 3).
- Additionally, a training part with Elapsed of only a few minutes is suspect even
  if all steps read COMPLETED — confirm real progress in
  `RUN_OUTPUT/output_<stage_run_id>_<jobid>.txt` before recording it as handled.

### 2. Classify each new terminal job
- **COMPLETED** (with python step verified OK in 1b) → record as handled. Nothing else.
- **CANCELLED** → cancelled by the user (or by this monitor in a previous step); record, no action.
- **TIMEOUT** → *normal* for chained training parts (`afterany` + `train_continue`).
  Record as handled. Exception: if it is the last part of the last stage of the pipeline
  (no dependent jobs existed), send an FYI email ("pipeline ended by timeout on final
  part — extend the chain if more training is wanted") but do **not** resubmit.
- **FAILED / NODE_FAIL / OUT_OF_MEMORY / BOOT_FAIL / PREEMPTED** → **CRASH** → step 3.

If several parts of the same run crashed in cascade (a failed part's dependents started
via `afterany` and failed too), treat them as **one** incident anchored on the earliest
failed job; mark all as handled.

### 3. On crash: cancel dependents, open incident, triage
1. **Cancel all remaining jobs of the run immediately** (afterany means they will run
   into the same problem otherwise) — always via the wrapper (raw scancel is denied):
   ```
   .claude/skills/job-monitor/cancel_run_jobs.sh <run_id>
   ```
   Record the cancelled job ids in the incident (so step 2 later classifies them as monitor-cancelled).
2. Create `INCIDENTS/<YYYY-MM-DD>_<run_id>_<jobid>/` with `incident.md` (timeline ledger,
   append every decision) and copy the crashed job's logs into `logs/` (full files — the
   run dir may be moved aside by a later `--restart-recopy`):
   `RUN_OUTPUT/output_<stage_run_id>_<jobid>.txt`, `RUN_LOGS/<stage_run_id>/{log.txt,error.txt}`
   (if present), and the SLURM `.err`/`.out` from RUN_LOGS.
3. **Triage** primarily from the tail of `output_<stage_run_id>_<jobid>.txt`
   (last ~300 lines — that is where tracebacks land), with `log.txt`/`.err`/`.out` as backup:
   - **Infra flake** — any of: NODE_FAIL/BOOT_FAIL state itself; NCCL timeout/unhandled
     system error; `uncorrectable ECC`; `srun: error: ... Socket timed out`;
     `slurmstepd: error: ... DUE TO NODE FAILURE`; `Stale file handle`;
     `Input/output error`; CUDA "unknown error"/driver error; preemption.
     → go straight to step 5 (plain resubmit, no code change). Short email.
   - **Code bug** — a Python traceback with a deterministic-looking exception
     (shape mismatch, KeyError, AttributeError, config error, NaN loss abort, assert),
     or the same infra-looking failure already seen for this run's same stage
     (signature = exception type + last stack frame; a repeat means it is not a flake).
     → step 4, then step 5 with `--restart-recopy`.
   - **OUT_OF_MEMORY**: treat as code/config bug (needs a config or code change), not flake.
   - Ambiguous → treat as code bug (investigation will settle it; the investigator may
     conclude "infra flake" in which case skip the fix stages).

### 4. Incident pipeline (code bugs only; at most ONE per pass, sequential)
Spawn each stage with the Agent tool, `run_in_background: false` and `model: "opus"`
(the monitor session runs on a cheaper model; the incident agents do the hard reasoning).
Prompts are templates in
this skill's `prompts/` dir — read them, fill the `{{...}}` placeholders, pass as the agent
prompt. Every stage writes its report into the incident dir.

1. **Investigator** (`prompts/investigator.md`) → writes `investigation.md`.
2. **Verifier** (`prompts/verifier.md`, fresh agent = clean context; gets raw logs +
   investigation.md) → writes `verification.md` with verdict CONFIRMED / REJECTED.
   - REJECTED once → rerun investigator with the objections appended; re-verify.
   - REJECTED twice → **escalate**: email diagnosis + disagreement, set run status
     `escalated`, stop (no fix, no resubmit).
   - Verifier may also conclude INFRA_FLAKE → skip to step 5 as a plain resubmit.
3. **Fixer** (`prompts/fixer.md`, spawn with `isolation: "worktree"`) → produces
   `fix.diff` (+ `fix_notes.md`, lint + relevant unit tests run inside the worktree).
4. **Fix-verifier** (`prompts/fix_verifier.md`, fresh agent) → `fix_verification.md`,
   verdict APPROVED / REJECTED. One rework round via the fixer, then escalate as above.
5. On APPROVED: apply the fix and commit, in whichever repo it actually touches — check the
   file paths in fix.diff, don't assume `$WGEN`. If a stage's config lives under
   `CONFIGS_REPO` (per JOBS_CSV's `config_dir` for this run) rather than `$WGEN/config/`,
   the fixer's `isolation: worktree` does **not** sandbox `CONFIGS_REPO` (only the primary
   `$WGEN` repo gets a worktree) — the fixer edits it live, in place. This is expected and
   pre-authorized: `git -C CONFIGS_REPO add/commit` the fix there with the same message
   convention (`[job-monitor] fix <run_id>: <one line>` + Claude co-author trailer), no need
   to pause and ask. It is still gated by the same fix-verifier APPROVED verdict as a
   `$WGEN` fix — that verdict is the actual safety gate, not which repo the diff lands in.
   If the working tree (either repo) has conflicting uncommitted changes in the same files,
   do NOT apply — escalate instead.

### 5. Resubmit (monitor only)
**Budget:** each run gets **2 automatic resubmits** (counter `auto_resubmits_used` in state,
counting flake and fix resubmits alike). Beyond that, every resubmit needs a **go**:
- Set run status `awaiting_approval`, send the approval email (template below), stop for this run.
- On a later pass, if `APPROVALS/<run_id>` exists (or the user approved in-session):
  delete the file, resubmit, increment counter, email confirmation.

**Launch flags:** the launcher records every live launch in `JOBS_CSV`
(columns: timestamp, run_id, pipeline_yaml, config_dir, stage_options (JSON list),
restart_from_stage, restart_recopy, job_ids, launch_cwd, argv — pipeline_yaml and
config_dir are absolute, so launch_cwd/argv are reference only). For a resubmit, look up
the run there:
- Take the run's most recent row for `pipeline_yaml` and `config_dir`; for `stage_options`,
  if the latest row is a restart with an empty list, fall back to the run's original
  (non-restart) row. Cache the result in state.
- Run not in JOBS_CSV (predates the recording): fall back to inferring the pipeline yaml —
  stage names are visible as `RUN_DIR/WeatherGenerator/config_command_line_<stage>.yaml`;
  find the yaml under `RUN_DIR/WeatherGenerator/config/` whose `stages:` list matches.
  Zero or >1 matches → escalate by email (ask which yaml) instead of guessing.
  No recorded `config_dir`/`stage_options` in this fallback: plain restarts are safe
  (the copy dir keeps the originals), but for a `--restart-recopy` a non-default
  `--config-dir` cannot be reconstructed → escalate rather than recopying with the
  default config dir.

**Command** (invoke by absolute path, pipeline_yaml/config_dir absolute from JOBS_CSV;
no `cd` — the launcher resolves its own paths):
```
$WGEN_PRIVATE/hpc/launch-slurm-sophie.py <pipeline_yaml> --run-id <run_id> --restart-from-stage <crashed stage> \
    [--config-dir <config_dir>] [--stage-options <opt> ...]
```
- Pass `--config-dir` iff the JOBS_CSV row has one; pass each recorded stage_options entry
  as a separate `--stage-options` flag. This matters most with `--restart-recopy`, which
  rebuilds the copied configs and `config_command_line_<stage>.yaml` files from scratch.
- Add `--restart-recopy` **only** when a code/config fix was applied (the launcher runs from a
  copied dir; without recopy the fix is invisible). Recopy refuses while jobs of the run are
  in squeue — dependents were already cancelled in step 3, but re-check squeue first.
- **Sibling stages, not just dependents, can be in squeue.** `--restart-from-stage X`
  resubmits every stage from X to the *end* of the passed pipeline yaml's `stages:` list
  (`active_stages = stages[start_idx:]` in the launcher) — not just X's own chain. If the
  yaml declares another stage after X that shares a parent with X (e.g. two sibling
  finetunes both `from: pretrain-ft`) but is *not* part of X's crash, a plain
  `--restart-from-stage X` would also relaunch that sibling — duplicating work and
  overwriting its checkpoints, even if it's healthy/already complete/still running.
  Before resubmitting, diff the crashed stage's position in `stages:` against the full
  list: if anything after it isn't actually downstream of the crash, write a truncated copy
  of the pipeline yaml (same file minus that stage's entry) into the incident dir and pass
  *that* to `--restart-from-stage`/`--restart-recopy` instead of the original. Dry-run
  reliably shows this scope (unaffected by the checkpoint-detection dry-run caveat below) —
  confirm only the intended stage's parts appear before going live. Never cancel or touch
  the sibling's own jobs to "solve" this — the truncated-yaml approach avoids needing to.
- **Dry-run cannot show whether the checkpoint-restart heuristic will fire.**
  `restart_has_checkpoint` defaults to `True` and the launcher's real detection
  (`_has_restart_checkpoints`) only runs `if not args.dry_run` — so dry-run always prints
  `FROM_RUN_ID=<run_id>-<stage>` regardless of whether a checkpoint actually exists; don't
  read anything into that field from a dry-run. What dry-run *does* reliably show is stage
  scope (previous bullet).
- **The checkpoint-restart heuristic itself can false-positive.** It only checks "does
  `models/<run_id>-<stage>/` contain anything," not "does a real `.chkpt` file exist." A
  stage that crashed before its first save (e.g. immediately, like the masking-assertion
  case) can still have a stray hyperparameter-dump JSON in that directory from init, which
  makes the launcher wrongly believe a checkpoint exists and try to `train_continue` from
  it — reproducing a `FileNotFoundError` on the *_latest.chkpt that was never written.
  Before a `--restart-recopy` resubmit, check `ls $SLURM_ROOT/../shared_work/models/<run_id>-<stage>/`
  (symlinked at `copy_wgen_dir/models`) for exactly this: if it's non-empty but contains no
  `.chkpt` file, move the stray file(s) aside into the incident dir's `moved_aside/`
  (reversible, backed up) so `_has_restart_checkpoints` correctly falls back to the upstream
  stage's checkpoint. This is pre-authorized — no need to pause and ask — but **scope it
  strictly to `models/<run_id>-<stage>/` for the run_id you are actively handling**; never
  touch another run's directory in that shared tree.
- Always run the **dry-run first** (default), sanity-check the printed sbatch commands
  (right stage, right run id), then rerun with `--no-dry-run`.
- Record new job ids in incident.md.

### 6. Email (one per incident, plus approval requests)
Send with this skill's helper (`/bin/mail` is broken on santis — no local sendmail;
the helper goes through smtp.cscs.ch):
```
.claude/skills/job-monitor/send_mail.py "<subject>" < email.txt
```
Save email.txt in the incident dir.
- Incident: subject `[wgen-monitor] <run_id> crashed at <stage>: <one-line cause>`.
  Body: what crashed (job id, stage, state, when) / triage or confirmed diagnosis /
  verifier verdict / fix summary + diff (or "plain resubmit — infra flake") /
  resubmit outcome + new job ids / resubmits used: N/2.
- Approval request: subject `[wgen-monitor] APPROVAL NEEDED: <run_id>`.
  Body: same summary + "To approve: `touch $WGEN/notes/job-monitor/approvals/<run_id>`
  on santis, or tell the monitor session 'approve <run_id>'."

### 7. Save state
Update `STATE`: `last_check`, per run: `handled_job_ids`, `auto_resubmits_used`,
`status` (healthy | incident_open | awaiting_approval | escalated | done),
`launch_flags` (pipeline_yaml, config_dir, stage_options — from JOBS_CSV),
`incidents` list, last failure signature per stage.
Append a one-line pass summary (jobs seen, incidents opened/resolved) via the
helper — never `echo "$(date ...)" >> ...`, which triggers a permission prompt:
```
.claude/skills/job-monitor/log_pass.sh "jobs_seen=... | incidents_opened=... incidents_resolved=..."
```
Then end the turn.

## Hard rules
- Never call sbatch/scancel/the launcher from a subagent — monitor only.
- Never record a train/inference job as COMPLETED without the step-level python
  check (step 1b) — the wrapper swallows srun failures.
- At most one incident pipeline per pass; at most 2 auto-resubmits per run, ever, without a go.
- Never resubmit when the new failure signature equals the previous one for that run+stage —
  that means the last resubmit/fix did not work → escalate.
- Escalation is always safe: email + `status: escalated` + do nothing until the user intervenes
  (user resets by editing state.yml or approving).
- Dry-run before every live launcher call. Report failures honestly in the email.
