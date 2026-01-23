L=10
N=3
sample=12 # Here `sample` denotes sample index.
for i in `seq 480 1 480`;
do 
    for j in `seq 0 1 20`;
        do cd ./ir_hf_L${L}_M${i}_N${N}_sample${sample}_${j}_mp/src/
        sleep 2
        ./submit_jobs.sh
        cd - # GO BACK TO THE ORIGINAL DIRECTORY.
        done
done
