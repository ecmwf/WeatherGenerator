#!/bin/bash
# This script is a test script for the WeatherGenerator
# Author: Ilaria Luise
# Date: $(date 2025-04-02)

TEST_ID=$(echo "$@" | grep -oP '(?<=--test_id )\S+')
if [ -z "$TEST_ID" ]; then
    read -p "Enter a name for this test: " TEST_ID
    if [ -z "$TEST_ID" ]; then
        echo "Error: TEST_ID cannot be empty."
        exit 1
    fi
fi

if [ -z "$WEATHERGEN_HOME" ]; then
    WEATHERGEN_HOME="./"
    echo "WEATHERGEN_HOME is not set. Defaulting to current directory: $WEATHERGEN_HOME"
else
    echo "WEATHERGEN_HOME is set to: $WEATHERGEN_HOME"
fi

set -e #stop if any of the following commands throws an error

uv run train --config "tests/config_test.py" --run_id $TEST_ID
uv run evaluate --run_id  $TEST_ID -start "2022-10-10" -end "2022-10-11" --samples 1000 --same_run_id --epoch 1
uv run pytest $WEATHERGEN_HOME/tests/test_metrics.py --run_id $TEST_ID

set +e