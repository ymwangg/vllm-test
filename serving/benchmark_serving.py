"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float

def apply_template(prompt):
    template = '[INST]<<SYS>>Please try to provide useful, helpful and actionable answers.<</SYS>>FIXME[/INST]'
    return template.replace("FIXME", prompt)

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int]]:
    # Load the dataset.
    with open(dataset_path) as fh:
        prompts = [json.loads(line)["prompt"] for line in fh.readlines()]
    prompt_token_ids = tokenizer(prompts).input_ids
    requests = [(apply_template(prompt), len(token_ids)) for prompt,token_ids in zip(prompts, prompt_token_ids)]
    # Sample the requests.
    sampled_requests = random.choices(requests, k=num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    per_token_latencies = []
    ttfts = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer.encode(outputs[i].generated_text))
            # print(f"Prompt: {outputs[i].prompt!r}")
            # print(f"Response: {outputs[i].generated_text!r}")
            total_output += output_len
            total_input += input_requests[i][1]
            per_token_latencies.append(outputs[i].latency / output_len)
            ttfts.append(outputs[i].ttft)
            completed += 1

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=total_output / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_tpot_ms=np.mean(per_token_latencies) * 1000,
        median_tpot_ms=np.median(per_token_latencies) * 1000,
        p99_tpot_ms=np.percentile(per_token_latencies, 99) * 1000,
    )

    return metrics


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    disable_tqdm: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    print(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print(f"Successful requests: {metrics.completed}")
    print(f"Benchmark duration: {benchmark_duration:2f} s")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
    print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
    print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_inthroughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ))

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        # file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        file_name = f"{args.output_name}"
        with open(file_name, "a+") as outfile:
            json.dump(result_json, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default model tokenizer.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max number of tokens to generate.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=0.5,
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disbale tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output.jsonl",
        help="Specify to save benchmark results to a json file",
    )
    args = parser.parse_args()
    main(args)

