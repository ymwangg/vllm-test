"""
This scripts show how to benchmark speculative decoding on a dataset.
"""
from vllm import LLM, SamplingParams
import argparse
import time
import json
import numpy as np
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    use_speculate: bool
    enforce_eager: bool
    model: str
    draft_model: str
    quantization: Optional[str]
    draft_model_quantization: Optional[str]
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    gpu_memory_utilization: float
    max_steps: int
    dataset: str
    output: str

    # engine configs
    batch_size: int
    tp_size: int
    draft_tp_size: int
    max_tokens: int
    speculate_length: int
    block_size: int

    # human_eval
    run_human_eval: bool

    @classmethod
    def from_cmd_args(cls):
        parser = argparse.ArgumentParser(
            description="Benchmark tool for speculative decoding in vLLM.")

        # Add command-line arguments
        parser.add_argument('--use-speculate',
                            action='store_true',
                            help='Use speculative decoding')
        parser.add_argument('--enforce-eager',
                            action='store_true',
                            help="Use cudagraph")
        parser.add_argument('--run-human-eval',
                            action="store_true",
                            help="Run human_eval benchmark")
        parser.add_argument('--model',
                            type=str,
                            default="codellama/CodeLlama-7b-Python-hf",
                            help='Model name')
        parser.add_argument('--draft-model',
                            type=str,
                            default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            help='Draft model name')
        parser.add_argument('--quantization', type=str, help='Quantization')
        parser.add_argument('--draft-model-quantization',
                            type=str,
                            help='Draft model quantization')
        parser.add_argument('--temperature',
                            type=float,
                            default=0.0,
                            help='Temperature')
        parser.add_argument('--top-p',
                            type=float,
                            default=1.0,
                            help='Top-p sampling cutoff')
        parser.add_argument('--top-k',
                            type=int,
                            default=-1,
                            help='Top-k sampling cutoff')
        parser.add_argument('--frequency-penalty',
                            type=float,
                            default=0.0,
                            help='Frequency penalty')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=0.75,
                            help='GPU memory utilization')
        parser.add_argument('--max-steps',
                            type=int,
                            default=-1,
                            help='Maximum number of steps')
        parser.add_argument('--dataset',
                            type=str,
                            default='dataset/humaneval.jsonl',
                            help='Dataset path')
        parser.add_argument('--output',
                            type=str,
                            default='output.jsonl',
                            help='Output path')
        parser.add_argument('--batch-size',
                            type=int,
                            default=1,
                            help='Batch size')
        parser.add_argument('--tp-size', type=int, default=2, help='TP size')
        parser.add_argument('--draft-tp-size',
                            type=int,
                            default=1,
                            help='Draft TP size')
        parser.add_argument('--max-tokens',
                            type=int,
                            default=512,
                            help='Maximum tokens')
        parser.add_argument('--speculate-length',
                            type=int,
                            default=5,
                            help='Speculate length')
        parser.add_argument('--block-size',
                            type=int,
                            default=16,
                            help='Block size')

        # Parse the command-line arguments and create an instance of BenchmarkConfig
        args = parser.parse_args()
        return cls(**vars(args))


def main(args: BenchmarkConfig):
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     frequency_penalty=args.frequency_penalty,
                                     max_tokens=args.max_tokens)

    # Sample prompts.
    with open(args.dataset) as fh:
        dataset = [json.loads(line) for line in fh.readlines()]

    # Prepare batches
    prompts_list = []
    step = 0
    offset = 0
    bs = args.batch_size
    while offset < len(dataset):
        if (args.max_steps >= 0 and step >= args.max_steps):
            break
        prompt = [item['prompt'] for item in dataset[offset:offset + bs]]
        prompts_list.append(prompt)
        step += 1
        offset += bs
        if len(prompt) < bs:
            break

    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp_size,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "quantization": args.quantization,
        "block_size": args.block_size,
    }
    if args.use_speculate:
        engine_kwargs.update({
            "draft_model":
            args.draft_model,
            "speculate_length":
            args.speculate_length,
            "draft_model_quantization":
            args.draft_model_quantization,
        })
    llm = LLM(**engine_kwargs)

    # warm up
    llm.generate(prompts_list[0], sampling_params)

    outputs_list = []
    t0 = time.monotonic()
    for step, prompts in enumerate(prompts_list):
        assert len(prompts) > 0
        t2 = time.monotonic()
        outputs = llm.generate(prompts, sampling_params)
        outputs_list.append(outputs)
        t3 = time.monotonic()
        num_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(f"batch_{step} throutput={num_tokens/(t3-t2)} tokens/sec")
    t1 = time.monotonic()

    prompt_lens = []
    text_lens = []
    history = []
    records = []
    for outputs in outputs_list:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            prompt_len = len(output.prompt_token_ids)
            prompt_lens.append(prompt_len)
            text_len = len(output.outputs[0].token_ids)
            text_lens.append(text_len)
            if args.use_speculate:
                history.append(output.outputs[0].acceptance_history)
                m = np.mean(history[-1])
            token_ids = output.prompt_token_ids + output.outputs[0].token_ids
            # Make sure eos is the last token if any.
            if 2 in token_ids:
                assert token_ids[-1] == 2
            records.append({
                'prompt': prompt,
                'response': generated_text,
                'acceptance': m if args.use_speculate else 0,
                'prompt_len': prompt_len,
                'response_len': text_len,
                'token_ids': token_ids
            })
    dt = t1 - t0
    print(f"Finished in {t1-t0} seconds")
    print(f"Mean avg prompt length = {np.mean(prompt_lens)}")
    print(f"Mean avg response length = {np.mean(text_lens)}")
    print(f"Throughput = {np.sum(text_lens)/dt} tokens/s")
    if args.use_speculate:
        avg_accept = np.mean([np.mean(x) for x in history if x])
        print(f"Avg accepted = {avg_accept}")
    with open(args.output, 'w') as fh:
        for item in records:
            fh.write(json.dumps(item) + '\n')

    if args.run_human_eval and args.dataset.endswith("humaneval.jsonl"):
        # To reproduce top@1 = 0.53 for CodeLlama34B-Python, run the following:
        #
        # python benchmark_speculate.py --batch-size 32 --run-human-eval --tp-size 8 \
        # --model codellama/CodeLlama-34b-Python-hf --use-speculate --max-tokens 512 \
        # --temperature 0.0 --frequency-penalty 0.1
        import os
        from human_eval.data import write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness
        # this is to stop tokenizers from printing lots of warnings
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        samples = [
            dict(task_id=raw_record['task_id'], completion=record['response'])
            for raw_record, record in zip(dataset, records)
        ]
        write_jsonl("tmp.jsonl", samples)
        results = evaluate_functional_correctness("tmp.jsonl")
        print(results)


if __name__ == "__main__":
    cmd_args = BenchmarkConfig.from_cmd_args()
    main(cmd_args)
