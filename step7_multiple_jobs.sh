L=10
N=3
for i in `seq 120 120 120`;
do 
    for j in `seq 2 1 2`;
        do cd ~/Yoshino2G3/ir_hf_L${L}_M${i}_N${N}_init${j}_mp/src/
        sbatch step7_job_tau_of_alpha.sh 
        done
done
