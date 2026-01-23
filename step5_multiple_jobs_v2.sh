L=10
N=3
for i in `seq 12 12 4`;
do 
    for j in `seq 2 1 2`;
        do cd ~/Yoshino4/ir_hf_L${L}_M${i}_N${N}_init${j}_mp/src/
        sbatch -x ea0211,ea0905,ea1106,ea1110,ea1210,eb0204,eb0907,eb1001,eb1009,eb1209,eb1211,eb1305,ec0407,ec0502,ec0503,ec0505,ec0905,ed0109 step5_job_overlaps_more_ave_over_init.sh
        done
done
#for i in `seq 1920 120 1920`;
for i in `seq 3840 120 3840`;
#for i in `seq 7680 120 7680`;
do 
    for j in `seq 2 1 2`;
        do cd ~/Yoshino4/ir_hf_L${L}_M${i}_N${N}_init${j}_mp/src/
        sbatch -x ea0211,ea0905,ea1106,ea1110,ea1210,eb0204,eb0907,eb1001,eb1009,eb1209,eb1211,eb1305,ec0407,ec0502,ec0503,ec0505,ec0905,ed0109 step5_job_overlaps_more_ave_over_init.sh
        done
done
