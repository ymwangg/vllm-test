tp=8
draft_tp=8
bs=64
max_tokens=256
max_steps=10
gpu_util=0.8

dataset=../dataset/humaneval.jsonl
target=/home/ubuntu/models/Meta-Llama-3-70B-Instruct
draft=/home/ubuntu/models/Meta-Llama-3-8B-Instruct

MPIRUN="mpirun --allow-run-as-root --bind-to none --mca btl_vader_single_copy_mechanism none --tag-output -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 -x FI_EFA_USE_DEVICE_RDMA=1 -x LD_LIBRARY_PATH -x PYTHONPATH -x MKL_DYNAMIC=FALSE"

CMD="$MPIRUN -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --gpu-memory-utilization $gpu_util"

$CMD --use-speculate --speculate-length 5 --run-profile

$CMD --run-profile
