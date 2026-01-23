L=10
N=3
sample=2
for i in `seq 480 1 480`;
do 
    for j in `seq 0 1 3`;
        do cd ir_hf_L${L}_M${i}_N${N}_sample${sample}_${j}_mp/src/
        ./step2_move_data.sh
        cd -
        done
done
