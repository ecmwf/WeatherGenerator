#!/bin/bash

# (from_run_id, to_run_id) tuples from mapping
run_pairs=(
    #"czfrhdae u2qk39pi"
    #"zha9i6x3 zswipf53"
    #"ypr1b3a4 dsdvzg59"
    #"vctfgruv r812ji96"
    #"htdwqjpx gj6eq2dx"
    #"gakr74pw ff80snum"
    #"qpogewjf v0yha29i"
    #"y3trwpx7 cbmk73y0"
    #"asnz2gyl ngdrjcbt"
    #"dd1cq6nv voulcvsi"
    #"dn15vfks urlp39xq"
    #"fl9xrpao ch1n05gd"
)

#for tuple in "${run_pairs[@]}"; do
#    read from_run_id run_id <<< "$tuple"
#    echo "From: $from_run_id → Run_id: $run_id"
#    sbatch weather_slurm_inferece.sh "$from_run_id" "$run_id"
#done





#for run_id in unov2gdz pv5hu3mc exsm2wty czfrhdae zha9i6x3 ypr1b3a4 vctfgruv htdwjqpx gakr74pw qpogewjf y3trwpx7 asnz2gyl dd1cq6nv dn15vfks fl9xrpao ; do
#for run_id in xqbky3ht whsolnr7 e0yzx968 ; do
for run_id in otn1u3oe r2z01faj xxjfcwq1 ; do
	echo $ "$run_id"
	sbatch weather_slurm_inferece.sh "$run_id"
done	
