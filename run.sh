#target=/home/ubuntu/models/CodeLlama-34b-Python-hf
target=/home/ubuntu/models/Llama-2-7b-chat-hf
draft=/home/ubuntu/models/draft_models/Suzuka-120M
#draft=/home/ubuntu/models/draft_models/TinyLlama-1.1B-Chat-v1.0
python benchmark_speculate.py --batch-size 32 --run-human-eval --tp-size 1 \
  --model $target --draft-model $draft --max-tokens 512 \
  --temperature 1.0 --top-p 0.9 --frequency-penalty 0.0 --block-size 16 --use-speculate 
