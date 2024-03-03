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
from collections import defaultdict
import os


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
    output_dir: str

    # engine configs
    batch_size: int
    tp_size: int
    draft_tp_size: int
    max_tokens: int
    speculate_length: int
    block_size: int

    # human_eval
    run_human_eval: bool

    # template
    use_system_prompt: bool
    system_template: str = '[INST]<<SYS>>Please try to provide useful, helpful and actionable answers.<</SYS>>PROMPT[/INST]'

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
        parser.add_argument(
            '--use-system-prompt',
            action="store_true",
            help="Use system prompt",
        )
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
                            default='dataset/mt.jsonl',
                            help='Dataset path')
        parser.add_argument('--output-dir',
                            type=str,
                            default='./output',
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


class Dataset:

    def __init__(self, path, llm, sampling_params, args):
        with open(path) as fh:
            prompts = [json.loads(line)["prompt"] for line in fh.readlines()]
        if args.use_system_prompt:
            assert "PROMPT" in args.system_template
            prompts = [
                args.system_template.replace("PROMPT", prompt)
                for prompt in prompts
            ]
        self.prompts = prompts
        self.llm = llm
        self.sampling_params = sampling_params
        self.args = args
        # directory [dataset, model_name, draft_model_name]
        base_output_dir = args.output_dir
        dataset_name = path.split("/")[-1].replace(".jsonl", "")
        model_name = os.path.normpath(args.model).split("/")[-1]
        self.dump_prefix = f"{base_output_dir}/{dataset_name}/{model_name}/"
        if args.use_speculate:
            draft_model_name = os.path.normpath(args.draft_model).split("/")[-1]
            self.dump_prefix += f"{draft_model_name}/"
        print(f"dumping to {self.dump_prefix}")
        os.makedirs(self.dump_prefix, exist_ok=True)

    def run(self, prompts):
        return self.llm.generate(prompts, self.sampling_params)

    def benchmark_bs(self, batch_size, max_steps=100):
        prompts = self.prompts
        # build bached requests
        prompts_list = []
        for step in range(0, len(prompts), batch_size):
            batched_prompts = prompts[step:step + batch_size]
            if len(batched_prompts) < batch_size:
                batched_prompts += prompts[:batch_size - len(batched_prompts)]
            prompts_list.append(batched_prompts)
        # warm up
        self.run(prompts_list[0])
        # benchmark
        outputs_list = []
        for step, prompts in enumerate(prompts_list[:max_steps]):
            t2 = time.monotonic()
            outputs = self.run(prompts)
            t3 = time.monotonic()
            outputs_list.append((outputs, t3 - t2))
            num_tokens = sum(
                len(output.outputs[0].token_ids) for output in outputs)
            print(f"batch_{step} throutput={num_tokens/(t3-t2)} tokens/sec")
        file_name = f"{self.dump_prefix}/BS_{batch_size}_Temperature_{self.args.temperature}_UseSysPrompt_{self.args.use_system_prompt}.jsonl"
        with open(file_name, 'w') as fh:
            for request_outputs, latency in outputs_list:
                num_tokens = sum(
                    len(request_output.outputs[0].token_ids)
                    for request_output in request_outputs)
                if self.args.use_speculate:
                    mean_acceptance = [
                        np.mean(request_output.outputs[0].acceptance_history)
                        for request_output in request_outputs
                    ]
                else:
                    mean_acceptance = None
                res = {
                    "acceptance_rate": mean_acceptance,
                    'latency': latency,
                    'throughput': num_tokens / latency,
                }
                fh.write(json.dumps(res))
                fh.write("\n")
        return outputs

    def benchmark_single(self, max_steps=100):
        # benchmark 1 request at a time
        # warm up
        self.run(self.prompts[0])
        outputs_list = []
        for step, prompt in enumerate(self.prompts[:max_steps]):
            t2 = time.monotonic()
            outputs = self.run([prompt])
            t3 = time.monotonic()
            outputs_list.append((outputs, t3 - t2))
            num_tokens = sum(
                len(output.outputs[0].token_ids) for output in outputs)
            print(f"batch_{step} throutput={num_tokens/(t3-t2)} tokens/sec")
        file_name = f"{self.dump_prefix}/BS_1_Temperature_{self.args.temperature}_UseSysPrompt_{self.args.use_system_prompt}.jsonl"
        with open(file_name, 'w') as fh:
            for request_outputs, latency in outputs_list:
                output = request_outputs[0].outputs[0]
                res = {
                    'prompt':
                    request_outputs[0].prompt,
                    'text':
                    output.text,
                    'acceptance_history':
                    output.acceptance_history,
                    "acceptance_rate":
                    np.mean(output.acceptance_history)
                    if output.acceptance_history else 0,
                    'latency':
                    latency,
                    'throughput':
                    len(output.token_ids) / latency,
                }
                fh.write(json.dumps(res))
                fh.write("\n")
        return outputs_list


class MTBenchDataset:

    def __init__(self,
                 path="dataset/mt_bench.jsonl",
                 use_system_prompt=False,
                 system_template=None):
        self.prompts_dict = defaultdict(list)
        with open(path) as fh:
            for line in fh.readlines():
                data = json.loads(line)
                self.prompts_dict[data['category']].append(data['turns'][0])
        if use_system_prompt:
            assert "PROMPT" in system_template
            for category, prompts in self.prompts_dict:
                self.prompts_dict[category] = [
                    system_template.replace("PROMPT", prompt)
                    for prompt in prompts
                ]


def main(args: BenchmarkConfig):
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     frequency_penalty=args.frequency_penalty,
                                     max_tokens=args.max_tokens)
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

    data = Dataset(args.dataset, llm, sampling_params, args)
    # outputs, dt = data.benchmark(32, 3, llm, sampling_params)
    # data.benchmark_single(max_steps=100)
    for batch_size in [1, 4, 8, 16, 32, 64]:
        data.benchmark_bs(batch_size, max_steps=100)


if __name__ == "__main__":
    cmd_args = BenchmarkConfig.from_cmd_args()
    main(cmd_args)
