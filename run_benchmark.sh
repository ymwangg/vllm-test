target=meta-llama/Llama-2-7b-hf
draft=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python benchmark_speculate.py --batch-size 1 --run-human-eval --tp-size 1 \
  --model $target --draft-model $draft --max-tokens 200 \
  --temperature 0.0 --top-p 1.0 --frequency-penalty 0.0 --block-size 16 --dataset long_ctx/gpt_3072.jsonl --max-steps 10 --use-speculate --speculate-length 5
