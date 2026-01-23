mkdir -p imag
mkdir -p imag/100samples

./job_mean_ener_in.sh
./job_mean_ener_layers.sh
./job_mean_ener_out.sh
./job_mean_ener.sh
