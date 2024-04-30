import json
import numpy as np
from vllm import LLM, SamplingParams

template = '[INST]<<SYS>>Please try to provide useful, helpful and actionable answers.<</SYS>>FIXME[/INST]'

# Sample prompts.
with open("gpt_3072.jsonl") as fh:
    prompts = [json.loads(l)['prompt'] for l in fh.readlines()]

prompts = [template.replace('FIXME',prompt) for prompt in prompts]
prompts = prompts[:4]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
target = "/home/ubuntu/models/Llama-2-7b-chat-hf"
#draft = "Doctor-Shotgun/smol_llama-220M-GQA-32k-linear"
draft = "Doctor-Shotgun/TinyLlama-1.1B-32k"

# Create an LLM.
llm = LLM(model=target,
          draft_model=draft,
          tensor_parallel_size=1,
          speculate_length=5)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    print(
        f"Prompt: {prompt!r}"
    )
    print(
        f"Response: {generated_text!r}"
    )
    print(f"Mean acceptance length={mean_num_accepted}")
