L=10
N=3
#Modify the W please (W is tw.) Before running this script, make sure in the Network.py the value of tw_list is of the same value as here (W).
W=1 
for i in `seq 120 1 120`;
do 
    for j in `seq 200 1 255`;
        do cd ~/Yoshino2G3_v3c/ir_hf_L${L}_M${i}_N${N}_sample${j}_mp/src/
        sbatch -x ea0211,ea0811,ea0905,ea1106,ea1110,ea1201,ea1210,eb0204,eb0907,eb0909,eb0912,eb1001,eb1009,eb1201,eb1209,eb1211,eb1305,ec0209,ec0202,ec0407,ec0502,ec0503,ec0505,ec0905,ed0109 step3b_job_autocorr.sh
        done
done
