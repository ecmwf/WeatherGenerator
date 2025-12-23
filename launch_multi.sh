
# # Resume training
# #for run_id in in9eslqf t5vqafju qz9n6815 scapqu18 yqy2ezoa bp9lcgwn vzeakjlb a8n4zrfs cxicf671 z3infogw a9hp2qju achvju39 cszpe803 y0kauh4s eneq4ahr ; do
# #for run_id in mia69x1h lgzkdwls jr39znm6 gzxgp7cw el6zytfd c64w3cgy m4x3a0jt manyrowd ijwbpy3k i9qkv084 l78tqy2z xn4wa7b2 s9sldzyb e29izt1j bmoc645w ; do
# for run_id in q4l8jb2e eytr9nki bbcm27x1 jjbfpuya wxehoqic sbylixor scguorkl uh0iz8sa jizcxg9f d2wgjec9 g10zvcn4 a2vlj964 z8vx03bg e3k2v450 qmil5gwk ; do
#   echo "$run_id"
#   #cp ../WeatherGenerator-private/hpc/santis/weathergen_slurm_train.sh /capstor/scratch/cscs/mkarlbau/slurm/slurm_weathergen_"$run_id"_dir/WeatherGenerator-private/hpc/santis/.
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
# done


# GRID SEARCH 1
# for lr in "5e-4" "1e-4" "5e-5" ; do
#   for w_dec in 0.05 0.1 0.2 0.4 0.6 ; do
#     echo "$lr $w_dec"
#     ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --chain-jobs 3 --link-venv --options lr_max=$lr weight_decay=$w_dec
#   done
# done


# GRID SEARCH 2
# beta1 0.6 0.7 0.8 0.9 0.95
# beta2 0.8 0.9 0.95, 0.99
# streams_directory="./config/streams/era5_1deg_w-aifs" "./config/streams/era5_1deg"
for beta1 in 0.6 0.7 0.8 0.9 0.95 ; do
  for beta2 in 0.8 0.9 0.95 0.99 ; do
    for sd in "./config/streams/era5_1deg_w-aifs" "./config/streams/era5_1deg" ; do
      echo "$beta1 $beta2 $sd"
      ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --chain-jobs 3 --link-venv --options lr_max=0.0001 weight_decay=0.1 adam_beta1=$beta1 adam_beta2=$beta2 streams_directory=$sd
  done
done