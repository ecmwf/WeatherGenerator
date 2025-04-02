#!/bin/bash
# This script is a test script for the WeatherGenerator
# Author: Ilaria Luise
# Date: $(date 2025-04-02)

set -e

TEST_ID=$(echo "$@" | grep -oP '(?<=--test_id )\S+')
if [ -z "$TEST_ID" ]; then
    read -p "Enter a name for this test: " TEST_ID
fi


uv run train --config "tests/config_test.py" --run_id $TEST_ID
uv run evaluate --run_id  $TEST_ID -start "2022-10-10" -end "2022-10-11" --samples 1000 --same_run_id
uv run pytest tests/test_metrics.py --run_id test --run_id $TEST_ID 