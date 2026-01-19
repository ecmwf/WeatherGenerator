
# GRID SEARCH 1
# for lr in "5e-4" "1e-4" "5e-5" ; do
#   for w_dec in 0.05 0.1 0.2 0.4 0.6 ; do
#     echo "$lr $w_dec"
#     ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --chain-jobs 3 --link-venv --options lr_max=$lr weight_decay=$w_dec
#   done
# done

# # Resume training
# #for run_id in in9eslqf t5vqafju qz9n6815 scapqu18 yqy2ezoa bp9lcgwn vzeakjlb a8n4zrfs cxicf671 z3infogw a9hp2qju achvju39 cszpe803 y0kauh4s eneq4ahr ; do
# #for run_id in mia69x1h lgzkdwls jr39znm6 gzxgp7cw el6zytfd c64w3cgy m4x3a0jt manyrowd ijwbpy3k i9qkv084 l78tqy2z xn4wa7b2 s9sldzyb e29izt1j bmoc645w ; do
# for run_id in q4l8jb2e eytr9nki bbcm27x1 jjbfpuya wxehoqic sbylixor scguorkl uh0iz8sa jizcxg9f d2wgjec9 g10zvcn4 a2vlj964 z8vx03bg e3k2v450 qmil5gwk ; do
#   echo "$run_id"
#   #cp ../WeatherGenerator-private/hpc/santis/weathergen_slurm_train.sh /capstor/scratch/cscs/mkarlbau/slurm/slurm_weathergen_"$run_id"_dir/WeatherGenerator-private/hpc/santis/.
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
# done

# # Delete validation.zarr and plots directories
# # ww0r248v, pg5oe6rq, jc10pgys, usizx285, yqy2ezoa, xbgazwtc, bxf4bdlz, tmnrdyk9, ue5ky698, hd4mvfa1, s301cqla, m63ocvtq, fq2jposb, fh1drqz8, r2ut4clp, u16fvide, t790hb8a, d0bou6tk, ge9zmby0, c745lzyr, sevmrclb, rqer37pc, ot3hqr0x, kj2qxw9k, x18rkx3s, afe4cwb0, srkhuy4g, mph51qok, bh2z0jkt, b0lwy3rk, qehytran, eionpvqj, oo4hq36z, a1x2cdf0, dn13x6ql, c7c480k2, uot4snvp, p3kvrg9j, mcugwbsp, qiz2bfkv, solj81d4, ku4r3omn, kro1j69u, gfm9e1z6, njhycz89
# for run_id in ww0r248v ; do
#   echo "results/$run_id/validation_chkpt00000_rank0000.zarr"
#   rm -r "results/$run_id/validation_chkpt00000_rank0000.zarr"
#   rm -r "results/$run_id/plots"
# done


# GRID SEARCH 2
# beta1 0.6 0.7 0.8 0.9 0.95
# beta2 0.8 0.9 0.95, 0.99
# streams_directory="./config/streams/era5_1deg, "./config/streams/era5_1deg_w-aifs""
# for beta1 in 0.6 0.7 0.8 0.9 0.95 ; do
#   for beta2 in 0.8 0.9 0.95 0.99 ; do
#     for sd in "./config/streams/era5_1deg" "./config/streams/era5_1deg_w-aifs" ; do
#       echo "$beta1 $beta2 $sd"
#       # ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --chain-jobs 3 --link-venv --options lr_max=0.0001 weight_decay=0.1 adam_beta1=$beta1 adam_beta2=$beta2 streams_directory=$sd
#       ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --chain-jobs 3 --link-venv --options lr_max=0.00005 weight_decay=0.05 adam_beta1=$beta1 adam_beta2=$beta2 streams_directory=$sd
#     done
#   done
# done

# # Train continue grid search 2
# # for run_id in s3cr2ef4 mrcw4e8h djuk9pzn aepv7wu1 v3h9k2bj ynlvs89z tz49cgl2 l98sz3jw m1k9eplh tgmftqi4 pu6e03ng lnph48zk t69trx2e j9eiz4dq z1sd0pnr tide3hv6 m45w7y1t np85vcnq ldv8n37w cfblam6s fb4ms5ec w584rjeg ctg1usw6 jz8tjv6m w4xqvt69 qrjo9xs0 u34liwsg exvmlypc uubfj3gz h392evts tp8b24cj oxfpqjw1 xpv8qnyd xpw5o9fx ti4vmpsg qkq9yjl7 e1enpxz3 nwj4z09v rhk794ou h1pwbi4a bcmvof51 b2weib8p hzd5uet7 k5pilh01 v4ah1kzx ppxufsjg pcl0snok wurb6xtk npmszy6v y9kfdpom x9ay3kmw irn3mjyc z4awp1g5 gmlzaqhj r0spjzik knouy46d q3tjv9di dwp6e7d9 n1qikpey j2dg98i0 zed62zhu a4vqxo9u suz4ilra m3vscjad mclsg8op rgtp5jy9 jnwrgzpf w9rkhg1v ob6r9moj t6ibtc1j mgyzswul z271maoe q57y4ve3 ohce4138 ao13xq8w t6bjgt9a x762dlfb xdo9plre hlrt72oe hbnm4u1x ; do
# # for run_id in z08sckyz k52e7hfo p09l6hpz gl3ev20k iaw2gbrn m6jniu34 obwocxim ep5q8gzn tkywi1hd lu3lj0z4 y75fha42 kmdx1g3n kmztan34 pg9k1fli qgv2layj heomgws2 yo8jyrbv jfd9lr0u x3rqmjes gk1ny5oh c0yf9pdt n2vm0sxq fvq6tyeo ql6spk9b p5ibvwda cp0k1an8 c01zraug hv97duk1 i2lrahz9 ge6v74ir e6ijt39q sx6m2ejs va9d6yv3 nw43c2n0 z1yi6s2c jsfhgdq1 d3o1a9lc u0gf6k3l gf3r9noc nb6342qh zyxq24l1 t5uh3gv9 hng8tw29 xd60xpu5 y90egrso jnsxo6mt lfq8om96 hhqnr8l9 sqne63ot unq1uwez cm3tolj1 k8irbcyk fqxcans2 ujqxld2f ksdn58ca wirp7n6q ttlvz8f2 xx68golp sm9i8lgx qeopqgju d2mj6l1d u1h675cf dge0wox1 gz6ad0sc g0hb53if n6nymxjc l1np8abi pybuh3o4 gn8jcdug vj6gn49y lnwequ80 t486blik il6f0p87 yixm5qls uafuiesw fh4vuolp ey17ezol im8g2pzi ff1ldwox wb47lhdj ; do 
# # for run_id in eosc19jg kl0rcua6 ukwujlf4 kfg9vle7 gp5cnl91 d5fwxbil lzg3iprw qxdtz5m1 p9d1ya0q x03dmoey f1pezdro i0jh85s1 gw6leyu4 ev2zrdsu b9kzho1t smgpkdbj tmk1adys wt89w7kp kuzwmra5 ezimdn39 npeywgca qx9bgzqw z8qvl037 jv6am9p8 lokz5def fvef6dpn gw3fz7rc f7jn9ep8 nblsm3uo ze1mh2cb t5fsvqxh vwz3ue4g ez2iolsm t946rtp8 bh65jf8k zb7s32vu momerjg2 o45qt3oc z8vwrchb lfc1rl63 gxibzqeo jx4pw8jb icnlv2kw lx5c0qh4 e2ndef59 aeqf5ozl e0riq4o6 psq79jhc cgncym6p idpvqbaw qn52qavo hygrvmh4 n6azvsmo difz8odj s5o06pyi v450oj3h g4n6iu78 yqtgkaz3 ezl2c8od vcophb7i kl7gop09 uhj4npb6 yjql2371 e1ungd2z awn0a856 hxtscki8 rln8xs45 i6jkuig9 s8qetp41 vn903brm sjxehz2y ch2cnwf1 qo7aihzs v2djkyu0 irmnhejc jct7zx6j y3ezyh1x prx0wqsk iecmarv3 kr4gtp38 ; do
# for run_id in eosc19jg kl0rcua6 ukwujlf4 kfg9vle7 gp5cnl91 d5fwxbil lzg3iprw qxdtz5m1 p9d1ya0q x03dmoey f1pezdro i0jh85s1 gw6leyu4 ev2zrdsu b9kzho1t smgpkdbj tmk1adys wt89w7kp kuzwmra5 ezimdn39 npeywgca qx9bgzqw z8qvl037 jv6am9p8 lokz5def fvef6dpn gw3fz7rc f7jn9ep8 nblsm3uo ze1mh2cb t5fsvqxh vwz3ue4g ez2iolsm t946rtp8 bh65jf8k zb7s32vu momerjg2 o45qt3oc z8vwrchb lfc1rl63 gxibzqeo jx4pw8jb icnlv2kw lx5c0qh4 e2ndef59 aeqf5ozl e0riq4o6 psq79jhc cgncym6p idpvqbaw qn52qavo hygrvmh4 n6azvsmo difz8odj s5o06pyi v450oj3h g4n6iu78 yqtgkaz3 ezl2c8od vcophb7i kl7gop09 uhj4npb6 yjql2371 e1ungd2z awn0a856 hxtscki8 rln8xs45 i6jkuig9 s8qetp41 vn903brm sjxehz2y ch2cnwf1 qo7aihzs v2djkyu0 irmnhejc jct7zx6j y3ezyh1x prx0wqsk iecmarv3 kr4gtp38 ; do
#   echo "$run_id"
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
# done


# # DROPOUT [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
# for dropout in 0.05 0.1 0.15 0.2 0.3 0.4 ; do
#   echo "$dropout"
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --link-venv --options lr_max=0.00005 weight_decay=0.05 adam_beta1=0.85 adam_beta2=0.9 streams_directory="./config/streams/era5_1deg_w-aifs" embed_dropout_rate=$dropout ae_local_dropout_rate=$dropout ae_adapter_dropout_rate=$dropout ae_global_dropout_rate=$dropout fe_dropout_rate=$dropout
# done

# # Train continue dropout
# for run_id in d2f1p4vh m8e7psdl n2nmxc7b flaucoz5 dnl5r61x aojt3c1z p1phw3g9 kacy7jbz uk0uvcfn d1fhev63 pxg7jnzt z40dbxjy fpymqrv3 pe93az4w saxqsfzb yjlzi5g7 zewh2o5n dy36qb7e ; do
#   echo "$run_id"
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
# done


# # NOISE LEVEL [1e-3, 5e-4, (1e-4), 5e-5, 1e-5]
# for nl in "1e-3" "5e-4" "1e-4" "5e-5" "1e-5" ; do
#   echo "$nl"
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --link-venv --options lr_max=0.00005 weight_decay=0.05 adam_beta1=0.85 adam_beta2=0.9 streams_directory="./config/streams/era5_1deg_w-aifs" impute_latent_noise_std=$nl
# done

# # Train continue noise level
# for run_id in fmpesclt h6lu3sh8 tgmwaifc vavdy4zf qf24wjsq nbc3il5x vvwizau9 wyhcr51m n207tod4 cey23p7w rh49o7yj qt72d4iy vl5n39cj ocyx09uw fydmc3vg ; do
#   echo "$run_id"
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
# done


# for run_id in  fmpesclt vvwizau9 vl5n39cj ; do
#   echo "$run_id"
#   ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
# done
#

for run_id in fl9xrpao ;  do
  echo "$run_id"
  ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --from-run-id $run_id --run-id $run_id --link-venv
done


# Cosine learning_rate test
#for lr_max in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" ; do
#  echo "$lr_max"
#  for from_run_id in dnl5r61x ; do
#	  ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --time 24:00:00 --from-run-id $from_run_id --link-venv --options istep=0 num_epochs=32 lr_max=$lr_max lr_policy_decay="cosine" forecast_steps=8 freeze_modules=".*global.*|.*local.*|.*adapter.*|.*ERA5.*"
#done
#done
