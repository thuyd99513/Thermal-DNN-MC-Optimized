L=10
N=3
M=480
STEP=1
sample=31

START=11
END=20
for i in `seq $M 1 $M`;
do 
    for j in `seq $START $STEP $END`;
        do cd ./ir_hf_L${L}_M${i}_N${N}_sample${sample}_${j}_mp/src
        #./step2_move_data.sh
        sbatch step3_job_overlap_J_in_and_out.sh
        cd -
        done
done
