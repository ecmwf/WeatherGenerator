#for run_id in unov2gdz pv5hu3mc exsm2wty czfrhdae zha9i6x3 ypr1b3a4 vctfgruv htdwjqpx gakr74pw qpogewjf y3trwpx7 asnz2gyl dd1cq6nv dn15vfks fl9xrpao ; do
for run_id in  lvlfd8er hr1l2whz a68hqu13 ;  do
	echo $ "$run_id"
	sbatch weather_slurm_inferece.sh "$run_id"
 done	
