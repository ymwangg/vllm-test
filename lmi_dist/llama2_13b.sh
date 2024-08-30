tp=1
draft_tp=1
bs=1
max_tokens=256
max_steps=10
gpu_util=0.8

dataset=../dataset/mt_bench.jsonl
target=/home/ubuntu/models/Llama-2-13b-chat-hf
draft=/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0


MPIRUN="mpirun --allow-run-as-root --bind-to none --mca btl_vader_single_copy_mechanism none --tag-output -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 -x FI_EFA_USE_DEVICE_RDMA=1 -x LD_LIBRARY_PATH -x PYTHONPATH -x MKL_DYNAMIC=FALSE"

for bs in 1;
do
CMD="$MPIRUN -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --use-speculate --speculate-length 5 --gpu-memory-utilization $gpu_util"
$CMD > $bs.txt 2>&1
CMD="$MPIRUN -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp"
$CMD > $bs.baseline.txt 2>&1
echo $bs
grep "Throughput" $bs.txt
grep "Throughput" $bs.baseline.txt
done
