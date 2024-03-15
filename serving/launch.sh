model=/home/ubuntu/models/Llama-2-7b-chat-hf
python -m vllm.entrypoints.api_server --model $model --disable-log-requests
