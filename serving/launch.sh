model=/home/ubuntu/models/Llama-2-70b-chat-hf
draft_model=/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0/
# draft_model=/home/ubuntu/models/Suzuka-73M
tp=4
common_args="--gpu-memory-utilization 0.8 --tensor-parallel-size $tp"
# Check if $1 matches something
if [ "$1" = "speculate" ]; then
    # Run code for option1
    echo "Running speculative decoding"
    port=8888
    python -m vllm.entrypoints.api_server --model $model --disable-log-requests --draft-model $draft_model --speculate-length 5 --port $port $common_args
    # Add your code here for option1
else
    # Default action when $1 does not match any expected value
    echo "Running normal decoding"
    port=8000
    python -m vllm.entrypoints.api_server --model $model --disable-log-requests --port $port $common_args
    echo "Usage: $(basename "$0") option1|option2"
    exit 1
fi
