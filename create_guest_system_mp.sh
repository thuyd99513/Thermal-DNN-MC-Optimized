L=10
N=3
T=100
R=24
W=1024
for i in `seq 120 1 120`;
do 
    for j in `seq 1 1 4`;
        do cp -r ir_hf_L_M_N_sample_mp ir_hf_L${L}_M${i}_N${N}_sample${j}_mp_tw${W}
        done
done
