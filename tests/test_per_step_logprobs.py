import torch
import numpy as np

from vllm import LLM, SamplingParams
from collections import defaultdict

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=512, logprobs=5)

# Create an LLM.
llm = LLM(model="lmsys/vicuna-13b-v1.5",
          draft_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",
          speculate_length=5)

num_requests = len(prompts)
for i in range(num_requests):
    prompt = prompts[i]
    llm._add_request(prompt, sampling_params, None)

# Run the engine.
request_cache = defaultdict(dict)
outputs = []
while llm.llm_engine.has_unfinished_requests():
    step_outputs = llm.llm_engine.step()
    for output in step_outputs:
        new_token_ids = []
        new_token_logprobs = []
        request_id = output.request_id
        assert len(output.outputs) == 1
        seq_output = output.outputs[0]
        if request_id not in request_cache:
            request_cache[request_id]['output_len'] = 0
            request_cache[request_id]['cum_logprob'] = 0
        prev_len = request_cache[request_id]['output_len']
        cur_len = len(seq_output.token_ids)
        request_cache[request_id]['output_len'] = cur_len
        new_token_ids = seq_output.token_ids[prev_len:cur_len]
        new_logprobs_list = seq_output.logprobs[prev_len:cur_len]
        new_logprobs = [
            logprobs[token_id]
            for token_id, logprobs in zip(new_token_ids, new_logprobs_list)
        ]
        prev_cum_logprob = request_cache[request_id]['cum_logprob']
        cur_cum_logprob = seq_output.cumulative_logprob
        request_cache[request_id]['cum_logprob'] = cur_cum_logprob
        delta_cum_logprobs = cur_cum_logprob - prev_cum_logprob
        ref_sum = sum(new_logprobs) if isinstance(new_logprobs[0], float) else sum([x.logprob for x in new_logprobs])
        torch.testing.assert_close(delta_cum_logprobs, ref_sum)
        print("-" * 100)
        print(f"request_id={request_id}, response=",
              list(zip(new_token_ids, new_logprobs)))
        assert len(new_token_ids) == len(new_logprobs) == cur_len - prev_len
        if output.finished:
            outputs.append(output)

# Sort the outputs by request ID.
# This is necessary because some requests may be finished earlier than
# its previous requests.
outputs = sorted(outputs, key=lambda x: int(x.request_id))

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Mean acceptance length={mean_num_accepted}"
    )
