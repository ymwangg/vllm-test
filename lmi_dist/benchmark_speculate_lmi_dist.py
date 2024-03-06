"""
This scripts show how to benchmark speculative decoding on a dataset.
"""
import torch
import argparse
import time
import json
import numpy as np
import vllm

from typing import List, Tuple

from vllm import SamplingParams, RequestOutput
from lmi_dist.engine import Engine
from lmi_dist.api import Request
from lmi_dist.init_engine import engine_from_args
from lmi_dist.comms import setup_torch_dist

use_speculate = True
run_humaneval = True
dataset_path = "../dataset/humaneval.jsonl"
batch_size = 1
max_steps = 512
output_file_name = "lmi_dist.json"
temperature = 0.0
top_p = 1.0
frequency_penalty = 0.0
max_tokens = 512

local_rank, world_size = setup_torch_dist()

def initialize_engine(args: argparse.Namespace) -> Engine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = vllm.EngineArgs.from_cli_args(args)
    engine = engine_from_args(engine_args)
    return engine

def generate(engine: Engine, test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    while test_prompts:
        prompt, sampling_params = test_prompts.pop(0)
        engine.add_request(
            Request(id=str(request_id), prompt=prompt, sampling_params=sampling_params))
        request_id += 1
    outputs: List[RequestOutput] = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs


def main(args):
    sampling_params = SamplingParams(temperature=temperature,
                                     top_p=top_p,
                                     frequency_penalty=frequency_penalty,
                                     max_tokens=max_tokens)

    # Sample prompts.
    with open(dataset_path) as fh:
        dataset = [json.loads(line) for line in fh.readlines()]

    # Prepare batches
    prompts_list = []
    step = 0
    offset = 0
    bs = batch_size
    while True:
        if offset >= len(dataset) or (max_steps >= 0
                                      and step >= max_steps):
            break
        prompt = [item['prompt'] for item in dataset[offset:offset + bs]]
        prompts_list.append(prompt)
        step += 1
        offset += bs
        if len(prompt) < bs:
            break

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=local_rank,
        init_method=None,
    )

    engine = initialize_engine(args)

    # warm up
    generate(engine, [(prompt, sampling_params) for prompt in prompts_list[0]])

    outputs_list = []
    t0 = time.monotonic()
    for step, prompts in enumerate(prompts_list):
        assert len(prompts) > 0
        t2 = time.monotonic()
        outputs = generate(engine, [(prompt, sampling_params) for prompt in prompts])
        outputs_list.append(outputs)
        t3 = time.monotonic()
        num_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(f"batch_{step} throutput={num_tokens/(t3-t2)} tokens/sec")
    t1 = time.monotonic()

    if local_rank > 0:
        return

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
            history.append(output.outputs[0].acceptance_history)
            m = np.mean(history[-1])
            token_ids = output.prompt_token_ids + output.outputs[0].token_ids
            records.append({
                'prompt': prompt,
                'response': generated_text,
                'acceptance': m,
                'prompt_len': prompt_len,
                'response_len': text_len,
                'token_ids': token_ids
            })
    dt = t1 - t0
    print(f"Finished in {t1-t0} seconds")
    print(f"Mean avg prompt length = {np.mean(prompt_lens)}")
    print(f"Mean avg response length = {np.mean(text_lens)}")
    print(f"Throughput = {np.sum(text_lens)/dt} tokens/s")
    if use_speculate:
        avg_accept = np.mean([np.mean(x) for x in history if x])
        print(f"Avg accepted = {avg_accept}")
    with open(output_file_name, 'w') as fh:
        json.dump(records, fh)

    if run_humaneval:
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
    # Create a sampling params object.
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    if use_speculate:
        import lmi_dist.patch.suzuka
    parser = vllm.EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
