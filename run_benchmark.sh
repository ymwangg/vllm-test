target=/home/ubuntu/models/CodeLlama-34b-Python-hf
draft=/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0
python benchmark_speculate.py --batch-size 32 --run-human-eval --tp-size 2 \
  --model $target --draft-model $draft --max-tokens 512 \
  --temperature 0.0 --top-p 1.0 --frequency-penalty 0.1 --block-size 16 --use-speculate
