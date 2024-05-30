# 1. Conduct tracking with multiprocessing in rotation speed
for revrot in 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025; do
python ../scripts/10_conduct_tracking.py 2017_Lan_aeqd_sample.nc --revrot $revrot --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 \
    -o=2017_Lan_ns7_nt1_rot${revrot}.nc --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 \
    --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5 &
done; wait

# 2-1. Finalize tracking
python ../scripts/20_finalize_tracking.py "2017_Lan_ns<ns>_nt1_rot<omega>.nc" \
    --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --ns=7 \
    --exclude stf stb score_ary psr -o 2017_Lan_ns7_nt1_ref.nc

# 2-2. Finalize tracking with reference flows
python ../scripts/20_finalize_tracking.py "2017_Lan_ns<ns>_nt1_rot<omega>.nc" \
    --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --ns=7 \
    --ref_flows=2017_Lan_ns7_nt1_ref.nc --tau_stri=24 --v_stri=20 --cth_stri=6 --omega_stri=0.0015 \
    --exclude stf stb score_ary psr -o=2017_Lan_ns7_nt1.nc

# 3. Plot AMVs at a specific time
python ../scripts/30_plot_velocity2d.py 2017_Lan_ns7_nt1.nc --it=24
## ==> see ./outputs/AMVs_it24.png
