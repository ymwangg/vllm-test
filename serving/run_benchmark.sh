# speculate 
common="--temperature 0.5 --top-p 0.9"
for rate in 0.2 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do
    for i in {1..3}; do
    num_prompts=$(echo "300 * $rate" | bc)
    num_prompts=$(printf "%.0f" "$num_prompts")
    echo "Running rate=$rate num_prompts=$num_prompts"
    python benchmark_serving.py --dataset ../dataset/mt.jsonl --model /home/ubuntu/models/Llama-2-70b-chat-hf/ $common --request-rate $rate --num-prompts $num_prompts --port 8000 --save-result --output-name baseline.jsonl
    python benchmark_serving.py --dataset ../dataset/mt.jsonl --model /home/ubuntu/models/Llama-2-70b-chat-hf/ $common --request-rate $rate --num-prompts $num_prompts --port 8888 --save-result --output-name speculate.jsonl
    done
done
