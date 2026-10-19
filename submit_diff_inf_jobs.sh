#!/usr/bin/env bash

# ==============================================================================
# Configuration
# ==============================================================================
START_DATE="2023-06-01T00:00"
END_DATE="2023-07-31T12:00"
STEP_HOURS=24

# Slurm queue throttling
MAX_JOBS=99             # Maximum allowed submitted/running jobs at a time
POLL_INTERVAL=300       # Check interval in seconds when queue limit is hit

RUN_ID_PREFIX="kz2n"
START_INDEX=0           # Starting suffix (0000)

# ==============================================================================
# Functions
# ==============================================================================
get_active_jobs() {
    # Returns the total number of jobs currently submitted/running for your user
    squeue --me -h | wc -l
}

# Convert dates to UTC epoch seconds for reliable date math
start_sec=$(date -u -d "${START_DATE}:00Z" +%s)
end_sec=$(date -u -d "${END_DATE}:00Z" +%s)
step_sec=$((STEP_HOURS * 3600))

curr_sec=$start_sec
job_idx=$START_INDEX

echo "Starting Slurm job submission loop..."
echo "Range: $START_DATE to $END_DATE ($STEP_HOURS-hour step)"
echo "Max concurrent queued jobs: $MAX_JOBS"
echo "----------------------------------------------------------------------"

# ==============================================================================
# Main Loop
# ==============================================================================
while [ "$curr_sec" -le "$end_sec" ]; do
    # Format current date ISO string and run_id
    curr_date=$(date -u -d "@$curr_sec" +"%Y-%m-%dT%H:%M")
    run_id=$(printf "%s%04d" "$RUN_ID_PREFIX" "$job_idx")

    # Queue limit throttling: wait until active jobs drop below MAX_JOBS
    active_jobs=$(get_active_jobs)
    while [ "$active_jobs" -ge "$MAX_JOBS" ]; do
        echo "[$(date +'%H:%M:%S')] Active Slurm jobs: $active_jobs >= $MAX_JOBS limit. Waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        active_jobs=$(get_active_jobs)
    done

    echo "Submitting job for date: $curr_date | run_id: $run_id"

    # Launch command
    ../WeatherGenerator-private/hpc/launch-slurm.py \
        --nodes 1 \
        --ntasks 4 \
        --gres gpu:4 \
        --ntasks-per-node 4 \
        --no-register \
        --stage inference \
        --wgen-dir . \
        --time 02:00:00 \
        --from-run-id mx7k88dy \
        --run-id "$run_id" \
        --options \
            test_config.samples_per_mini_epoch=4 test_config.output.num_samples=4 \
            data_loading.num_workers=0 validation_config.validation_noise_levels=[] \
            diffusion_rollout=True fe_diffusion_num_ensemble_members=4 \
            decode_members_per_pass=1 decode_cells_per_pass=4 \
            test_config.forecast.chunk_size=1 test_config.compute_full_validation_loss=False \
            training_config.losses='{"physical": {"type": "LossPhysical", "weight": 0.0, "loss_fcts": {"mse": {}}, "target_and_aux_calc": "Physical"}}' \
            load_decoder_chkpt.run_id=kz2n1t60 load_decoder_chkpt.mini_epoch=-1 \
            test_config.shuffle=False test_config.losses='{"physical": {"type": "LossPhysical", "weight": 0.0, "loss_fcts": {"mse": {}}, "target_and_aux_calc": "Physical"}}' \
            test_config.model_input.forecasting.num_steps_input=1 test_config.forecast.offset=1 \
            diffusion_particle_guidance=True diffusion_particle_guidance_strength=0.2 \
            diffusion_particle_guidance_sigma_power=1.0 diffusion_particle_guidance_kernel_space=x0 \
            diffusion_particle_guidance_cell_chunk=4096 diffusion_particle_guidance_fstep_decay=3.0 \
            diffusion_conditioning_noise_std=0.05 diffusion_conditioning_noise_fstep_decay=2 \
            diffusion_conditioning_noise_norm_scope=token diffusion_conditioning_noise_schedule=fixed \
            training_config.forecast.num_steps=40 \
            test_config.start_date="$curr_date" test_config.end_date=2023-12-31T00:00

    # Advance time and run index
    curr_sec=$((curr_sec + step_sec))
    job_idx=$((job_idx + 1))
done

echo "----------------------------------------------------------------------"
echo "All jobs successfully submitted!"