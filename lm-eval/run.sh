#gsm8k,mmlu
tasks=gsm8k,mmlu
tasks=gsm8k
target=/home/ubuntu/models/Llama-2-70b-chat-hf
#target=/home/ubuntu/models/draft_models/Suzuka-120M
#target=/home/ubuntu/models/draft_models/TinyLlama-1.1B-Chat-v1.0
draft=/home/ubuntu/models/draft_models/Suzuka-120M

# Huggingface Reference
#accelerate launch -m lm_eval --model hf \
#    --model_args pretrained=$target \
#    --tasks $tasks \
#    --device cuda:0 \
#    --batch_size 8

lm_eval --model vllm \
    --model_args pretrained=$target,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,draft_model=$draft\
    --tasks $tasks \
    --batch_size 64

lm_eval --model vllm \
    --model_args pretrained=$target,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1\
    --tasks $tasks \
    --batch_size 64
