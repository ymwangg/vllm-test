tp=8
draft_tp=8
bs=64
max_tokens=256
max_steps=10
gpu_util=0.8

CMD="mpirun --allow-run-as-root --bind-to none --mca btl_vader_single_copy_mechanism none --tag-output -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 -x FI_EFA_USE_DEVICE_RDMA=1 -x LD_LIBRARY_PATH -x PYTHONPATH -x MKL_DYNAMIC=FALSE"

dataset=../dataset/humaneval.jsonl
target=/home/ubuntu/models/Llama-3-70b-instruct-hf
draft=/home/ubuntu/models/Llama-3-8b-instruct-hf

# no vmmtensor + no custom-ar
$CMD -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --run-human-eval --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --use-speculate --speculate-length 5 --gpu-memory-utilization $gpu_util --disable-custom-all-reduce

# vmmtensor + custom-ar
$CMD -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --run-human-eval --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --use-speculate --speculate-length 5 --gpu-memory-utilization $gpu_util --disable-custom-all-reduce --vmmtensor-kv-cache

# no vmmtensor + custom-ar
$CMD -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --run-human-eval --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --use-speculate --speculate-length 5 --gpu-memory-utilization $gpu_util

# vmmtensor + custom-ar
$CMD -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --run-human-eval --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --use-speculate --speculate-length 5 --gpu-memory-utilization $gpu_util --vmmtensor-kv-cache
