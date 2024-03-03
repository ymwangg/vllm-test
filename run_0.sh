export CUDA_VISIBLE_DEVICES=0,1,2,3
tp_sizes=(2 4 4)
target_names=(CodeLlama-34b-Python-hf Llama-2-70b-chat-hf)
draft_names=(Suzuka-73M Suzuka-120M TinyLlama-1.1B-Chat-v1.0)
dataset="./dataset/mt.jsonl"
output_dir="./output"

for ((i=0; i<${#target_names[@]}; i++)); do
    target_name=${target_names[$i]}
    target=/home/ubuntu/models/$target_name
    tp=${tp_sizes[$i]}
    echo "Running $target"
    # none speculate case
    python benchmark.py --tp-size $tp --model $target --dataset $dataset --output-dir $output_dir --max-tokens 512 --temperature 0.0 --top-p 1.0 --frequency-penalty 0.0 --block-size 16 --use-system-prompt
    for ((j=0; j<${#draft_names[@]}; j++)); do
        draft_name=${draft_names[$j]}
        draft=/home/ubuntu/models/$draft_name
        echo "Running $target $draft"
        python benchmark.py --tp-size $tp --model $target --draft-model $draft --dataset $dataset --output-dir $output_dir --max-tokens 512 --temperature 0.0 --top-p 1.0 --frequency-penalty 0.0 --block-size 16 --use-system-prompt --use-speculate 
    done
done
