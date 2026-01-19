#!/bin/bash

# Extract all unique (lr, from_run_id) tuples from mapping
tuples=(
    "5e-4 dnl5r61x"    # zd4t0zmp
    "5e-4 vvwizau9"    # fzmgdsev  
    "5e-4 wyhcr51m"    # rqhn8y14
    "5e-5 dnl5r61x"    # nibpxofg
    "5e-5 wyhcr51m"    # zylxr8pm
    "1e-5 dnl5r61x"    # s1urb38z
    "1e-5 vvwizau9"    # lmix3abo
    "1e-5 wyhcr51m"    # gu20n5l8
    "5e-6 dnl5r61x"    # sxlqdhue
    "5e-6 vvwizau9"    # s8pfvmle
)

echo "Launching ${#tuples[@]} experiments..."

for tuple in "${tuples[@]}"; do
    read lr_max from_run_id <<< "$tuple"
    echo "=== $from_run_id @ $lr_max ==="
    
    ../WeatherGenerator-private/hpc/launch-slurm.py \
        --nodes 2 \
        --time 24:00:00 \
        --from-run-id "$from_run_id" \
        --link-venv \
        --options istep=0 num_epochs=32 lr_max=$lr_max lr_policy_decay="cosine" forecast_steps=8 freeze_modules=".*global.*|.*local.*|.*adapter.*|.*ERA5.*"
    
    echo "----------------------------------------"
done

echo "All $((${#tuples[@]})) jobs submitted!"

