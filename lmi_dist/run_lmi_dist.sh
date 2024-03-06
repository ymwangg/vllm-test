tp=4
mpirun -n $tp --allow-run-as-root --mca btl ^openib --mca btl_tcp_if_include eth0 -x UCX_TLS=tcp -x SUZUKA=true python  benchmark_speculate_lmi_dist.py --tensor-parallel-size $tp --model /home/ubuntu/models/CodeLlama-34b-Python-hf/ --draft-model /home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0
