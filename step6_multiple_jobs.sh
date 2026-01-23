L=10
N=3
for i in `seq 120 120 240`;
do 
    for j in `seq 2 1 2`;
        do cd ~/Yoshino2G3/ir_hf_L${L}_M${i}_N${N}_sample${j}_mp/src/
        sbatch -x ea0211,ea0905,ea1106,ea1110,ea1210,eb0204,eb0907,eb1001,eb1009,eb1209,eb1211,eb1305,ec0407,ec0502,ec0503,ec0505,ec0905,ed0109 step6_job_relaxation_time.sh 
        done
done
for i in `seq 480 480 960`;
do 
    for j in `seq 2 1 2`;
        do cd ~/Yoshino2G3/ir_hf_L${L}_M${i}_N${N}_sample${j}_mp/src/
        sbatch -x ea0211,ea0905,ea1106,ea1110,ea1210,eb0204,eb0907,eb1001,eb1009,eb1209,eb1211,eb1305,ec0407,ec0502,ec0503,ec0505,ec0905,ed0109 step6_job_relaxation_time.sh 
        done
done
for i in `seq 1920 1920 1920`;
do 
    for j in `seq 2 1 2`;
        do cd ~/Yoshino2G3/ir_hf_L${L}_M${i}_N${N}_sample${j}_mp/src/
        sbatch -x ea0211,ea0905,ea1106,ea1110,ea1210,eb0204,eb0907,eb1001,eb1009,eb1209,eb1211,eb1305,ec0407,ec0502,ec0503,ec0505,ec0905,ed0109 step6_job_relaxation_time.sh 
        done
done
