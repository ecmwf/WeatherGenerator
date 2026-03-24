#!/bin/bash
#SBATCH --qos=nf
#SBATCH --job-name=squeue_dump
#SBATCH --time=00:05:00

squeue --all --format="%.18i %.12j %.10u %.8T %.10M %.6D %.20R"