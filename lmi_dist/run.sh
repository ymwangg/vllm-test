tp=1
draft_tp=1
bs=1
max_tokens=256
max_steps=3
gpu_util=0.8

dataset=humaneval.jsonl
target=/home/ubuntu/models/llama-3-8b-instruct-awq
draft=/home/ubuntu/models/llama-3-8b-instruct-awq


MPIRUN="mpirun --allow-run-as-root --bind-to none --mca btl_vader_single_copy_mechanism none --tag-output -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 -x FI_EFA_USE_DEVICE_RDMA=1 -x LD_LIBRARY_PATH -x PYTHONPATH -x MKL_DYNAMIC=FALSE"

for bs in 64;
do
CMD="$MPIRUN -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp  --use-speculate --speculate-length 5 --gpu-memory-utilization $gpu_util"
#CMD="$MPIRUN -n $tp python  benchmark_speculate.py --tp-size $tp --model $target --draft-model $draft --batch-size $bs --dataset $dataset --max-steps $max_steps  --max-tokens $max_tokens --draft-tp-size $draft_tp"
#echo "no vmmtensor + no custom-ar"
#$CMD --disable-custom-all-reduce > 0.txt 2>&1
#grep "Throughput" 0.txt

#echo "vmmtensor + custom-ar"
#$CMD --disable-custom-all-reduce --vmmtensor-kv-cache > 1.txt 2>&1
#grep "Throughput" 1.txt

# echo "no vmmtensor + custom-ar"
echo $bs
$CMD > $bs.txt 2>&1
grep "Throughput" 2.txt
done

#echo "vmmtensor + custom-ar"
#$CMD --vmmtensor-kv-cache > 3.txt 2>&1
#grep "Throughput" 3.txt
