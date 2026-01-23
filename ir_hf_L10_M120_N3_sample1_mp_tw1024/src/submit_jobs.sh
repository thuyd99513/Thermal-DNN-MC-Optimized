#rm -rf ../data/*

index=0
L=10
N=3
Nin=784
Nout=2
S=1200
#tw=0 # If one want to run simulations for all tw in a node, then they do not need tw as an input.

C=2
M=120

sbatch -x ea0211,ea0905,ea1106,ea1110,ea1210,eb0204,eb0907,eb1001,eb1009,eb1209,eb1211,eb1305,ec0202,ec0407,ec0502,ec0503,ec0505,ec0514,ec0905,ec0907,ec1102,ed0109 step1_job_host_guest.sh 66.7 $C $Nin $Nout $L $M $N $S $index
