L=10
N=3
sample=31
M=480
START=11
END=20
#Modify the W please (W is tw.) Before running this script, make sure in the Network.py the value of tw_list is of the same value as here (W).
for i in `seq $M 1 $M`;
do 
    for j in `seq $START 1 $END`;
        do cd ir_hf_L${L}_M${i}_N${N}_sample${sample}_${j}_mp/src/
        ./step2_move_data.sh
        sbatch step3_job_overlap.sh
        cd -
        done
done
