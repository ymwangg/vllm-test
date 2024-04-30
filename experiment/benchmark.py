"""
This scripts show how to benchmark speculative decoding on a dataset.
"""
from vllm import LLM, SamplingParams
import time
import json
import numpy as np
import argparse
import os
import yaml
import random

from fastchat.model.model_adapter import get_conversation_template
from dataclasses import dataclass, fields
from collections import defaultdict
from typing import List

random.seed(0)


@dataclass
class Config:
    config: str
    model: str
    draft_model: str

    shuffle_data: bool = False
    dataset: str = "../dataset/humaneval.jsonl"

    use_chat_template: bool = True
    system_prompt: str = "Please try to provide useful, helpful and actionable answers."

    use_speculate: bool = True
    speculate_length: int = 5
    enforce_eager: bool = False
    tp_size: int = 1

    gpu_memory_utilization: float = 0.8

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    frequency_penalty: float = 0.0
    max_tokens: int = 256

    @classmethod
    def from_yaml(cls, yaml_data: dict) -> 'Config':
        return cls(**yaml_data)

    @classmethod
    def from_yaml_file(cls, path: str) -> 'Config':
        with open(path) as fh:
            yaml_data = yaml.safe_load(fh)
        yaml_data['config'] = os.path.abspath(path)
        return cls.from_yaml(yaml_data)

    def to_yaml(self) -> str:
        yaml_dict = {}
        for field in fields(self):
            yaml_dict[field.name] = getattr(self, field.name)
        return yaml.dump(yaml_dict, sort_keys=False)


class Dataset:

    def __init__(self, config):
        self.config = config
        with open(config.dataset) as fh:
            rawdata = [json.loads(line)["prompt"] for line in fh.readlines()]
        if config.use_chat_template:
            self.data = []
            for prompt in rawdata:
                template = get_conversation_template(config.model)
                template.set_system_message(config.system_prompt)
                template.append_message(template.roles[0], prompt)
                template.append_message(template.roles[1], None)
                self.data.append(template.get_prompt())
        else:
            self.data = rawdata
        self.index = 0

    def reset(self):
        self.index = 0

    def fetch_next(self, shuffle_data=False):
        if shuffle_data:
            return random.choice(self.data)
        result = self.data[self.index % len(self.data)]
        self.index += 1
        return result


class Executor:

    def __init__(self, config):
        self.config = config
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            frequency_penalty=config.frequency_penalty,
            max_tokens=config.max_tokens)
        self.dataset = Dataset(config)
        self.llm = self.load_engine(config)
        self.outputs = defaultdict(dict)
        config_file_name = config.config.split("/")[-1]
        self.full_output = config_file_name.replace(".yaml", ".full.jsonl")
        self.summary_output = config_file_name.replace(".yaml",
                                                       ".summary.jsonl")
        for fname in [self.full_output, self.summary_output]:
            if os.path.exists(fname):
                os.remove(fname)

    def reset(self):
        self.outputs.clear()
        self.dataset.reset()

    def load_engine(self, config):
        engine_kwargs = {
            "model": config.model,
            "tensor_parallel_size": config.tp_size,
            "enforce_eager": config.enforce_eager,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if config.use_speculate:
            engine_kwargs.update({
                "draft_model": config.draft_model,
                "speculate_length": config.speculate_length,
            })
        llm = LLM(**engine_kwargs)
        return llm

    def add_request(self, next_request):
        llm = self.llm
        llm_engine = llm.llm_engine
        t0 = time.time()
        llm._add_request(next_request, self.sampling_params, None)
        request_id = llm_engine.scheduler.waiting[-1].request_id
        assert request_id not in self.outputs
        record = self.outputs[request_id]
        record['request_id'] = request_id
        record['t0'] = t0

    def finish_request(self, output, batch_size):
        request_id = output.request_id
        assert request_id in self.outputs
        record = self.outputs[request_id]
        response = output.outputs[0]
        record['prompt'] = output.prompt
        record['text'] = response.text
        record['num_output_tokens'] = len(response.token_ids)
        t1 = time.time()
        record['t1'] = t1
        dt = t1 - record['t0']
        record['mean_tpot_ms'] = dt / len(response.token_ids) * 1000
        record['mean_throughput'] = len(response.token_ids) / dt
        record['batch_size'] = batch_size
        if self.config.use_speculate:
            record['mean_accept_length'] = np.mean(response.acceptance_history)
        print(f"request_id={request_id} batch_size={batch_size} "
              f"num_tokens={record['num_output_tokens']} "
              f"mean_tpot_ms={record['mean_tpot_ms']:.2f} "
              f"mean_throughput={record['mean_throughput']:.2f}")

    def profile(self, batch_size=1, time_limit_s=10, request_limit=None):
        """_summary_

        Args:
            batch_size (int, optional): batch size. Defaults to 1.
            time_limit_s (int, optional): max elapsed time to sample response data. Defaults to 10 seconds.
            request_limit (_type_, optional): max number of requests to sampled. Defaults to None.
                Cannot be used with time_limit_s together.
        """
        assert time_limit_s is not None and request_limit is None or \
               time_limit_s is None and request_limit is not None, "time_limit and request_limit cannot be both specified at the same time"
        num_added = 0
        # constant batch size
        t0 = time.time()
        for _ in range(batch_size):
            next_request = self.dataset.fetch_next(config.shuffle_data)
            if not next_request:
                break
            num_added += 1
            self.add_request(next_request)
        llm = self.llm
        llm_engine = llm.llm_engine
        if time_limit_s is not None:
            while (time.time() - t0) < time_limit_s:
                # saturate the worker with max batch sizes
                while llm_engine.get_num_unfinished_requests() < batch_size:
                    next_request = self.dataset.fetch_next(config.shuffle_data)
                    num_added += 1
                    self.add_request(next_request)
                # run decoding step
                step_outputs = llm.llm_engine.step()
                # record finished request
                for output in step_outputs:
                    # Only requests finished within simulation time are recorded
                    if output.finished:
                        self.finish_request(output, batch_size)
                t1 = time.time()
        else:
            while num_added < request_limit or llm_engine.has_unfinished_requests(
            ):
                # saturate the worker with max batch sizes
                while llm_engine.get_num_unfinished_requests(
                ) < batch_size and num_added < request_limit:
                    next_request = self.dataset.fetch_next(config.shuffle_data)
                    num_added += 1
                    self.add_request(next_request)
                # run decoding step
                step_outputs = llm.llm_engine.step()
                # record finished request
                for output in step_outputs:
                    # Only requests finished within simulation time are recorded
                    if output.finished:
                        self.finish_request(output, batch_size)
                t1 = time.time()
        dt = t1 - t0
        # Finish all unfinished requests
        for unfinished_requests in llm_engine.scheduler.waiting + llm_engine.scheduler.running:
            request_id = unfinished_requests.request_id
            self.outputs.pop(request_id)
            llm_engine.abort_request(request_id)
        assert not llm_engine.has_unfinished_requests()
        assert len(
            self.outputs) > 0, "Too few requests finished within time limits"
        # dump the results
        records = list(self.outputs.values())
        total_num_tokens = sum(record['num_output_tokens']
                               for record in records)
        global_throughput = total_num_tokens / dt
        mean_tpot_ms = np.mean([record['mean_tpot_ms'] for record in records])
        mean_throughput = np.mean(
            [record['mean_throughput'] for record in records])
        global_results = {
            "batch_size": batch_size,
            "num_samples": len(records),
            "global_throughput": global_throughput,
            "mean_tpot_ms": mean_tpot_ms,
            "mean_throughput": mean_throughput,
            "use_speculate": config.use_speculate,
        }
        if config.use_speculate:
            mean_acceptance = np.mean(
                [record['mean_accept_length'] for record in records])
            global_results['mean_acceptance'] = mean_acceptance

        with open(self.summary_output, 'a+') as fh:
            fh.write(json.dumps(global_results))
            fh.write("\n")

        with open(self.full_output, 'a+') as fh:
            for record in records:
                fh.write(json.dumps(record))
                fh.write("\n")


def main(config: Config):
    executor = Executor(config)
    batch_sizes = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    time_limit = 5 * 60  # simulate 5 minutes
    for bs in batch_sizes:
        executor.profile(bs, time_limit_s=time_limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for vllm")
    parser.add_argument("config", help="Path to yaml config file")
    config = parser.parse_args()
    config = Config.from_yaml_file(config.config)
    main(config)